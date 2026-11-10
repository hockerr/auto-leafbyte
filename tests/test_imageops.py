"""Unit tests for the Lead-owned image primitives (imageops)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from leafbyte import imageops  # noqa: E402
from leafbyte.config import Config  # noqa: E402


class LumaOtsuTest(unittest.TestCase):
    def test_luma_matches_weights(self) -> None:
        img = np.zeros((1, 3, 3), dtype=np.uint8)
        img[0, 0] = (255, 0, 0)
        img[0, 1] = (0, 255, 0)
        img[0, 2] = (0, 0, 255)
        values = imageops.luma(img)[0]
        self.assertAlmostEqual(values[0], 0.299, places=6)
        self.assertAlmostEqual(values[1], 0.587, places=6)
        self.assertAlmostEqual(values[2], 0.114, places=6)

    def test_otsu_splits_synthetic_bimodal_image(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:50] = 30    # dark half
        image[50:] = 220   # bright half
        threshold = imageops.suggested_threshold(image)
        self.assertGreater(threshold, 30 / 255)
        self.assertLess(threshold, 220 / 255)
        mask = imageops.binarize(image, threshold, "white")
        self.assertTrue(mask[:50].all())
        self.assertFalse(mask[50:].any())

    def test_binarize_black_background(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[0, 0] = 255
        mask = imageops.binarize(image, 0.5, "black")
        self.assertTrue(mask[0, 0])
        self.assertEqual(1, int(mask.sum()))

    def test_threshold_is_strict(self) -> None:
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        image[0, 0] = 128  # luma = 0.502
        threshold = 128 / 255
        self.assertFalse(imageops.binarize(image, threshold, "white")[0, 0])
        self.assertTrue(imageops.binarize(image, 0.51, "white")[0, 0])

    def test_histogram_diagnostics_are_bounded(self) -> None:
        rng = np.random.default_rng(7)
        flat = np.full((80, 80, 3), 200, dtype=np.uint8)
        flat[:40] = 40
        flat = np.clip(flat.astype(np.int16) + rng.integers(-4, 5, flat.shape), 0, 255).astype(np.uint8)
        stats = imageops.threshold_diagnostics(flat, imageops.suggested_threshold(flat))
        for key, value in stats.items():
            self.assertGreaterEqual(value, 0.0, key)
            self.assertLessEqual(value, 1.0, key)
        self.assertGreater(stats["separation"], 0.8)


class WarpTest(unittest.TestCase):
    def test_rectified_size_uses_bounding_box(self) -> None:
        quad = [(10, 20), (210, 25), (205, 220), (15, 215)]
        # upstream: S = min(1200, floor(min(bbox_w, bbox_h))) = min(200, 200)
        self.assertEqual(200, imageops.rectified_size_for_quad(quad, 1200))
        # ... but the 1200 cap wins when the quad is bigger
        self.assertEqual(100, imageops.rectified_size_for_quad(quad, 100))

    def test_warp_quad_maps_centre(self) -> None:
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[90:110, 90:110] = 255  # 20x20 white patch centred on (100, 100)
        quad = [(20, 20), (180, 20), (180, 180), (20, 180)]
        warped = imageops.warp_quad(image, quad, 80)
        self.assertEqual((80, 80, 3), warped.shape)
        self.assertGreater(warped[38:42, 38:42].mean(), 240)   # patch landed at the centre
        self.assertLess(warped[0:6, 0:6].mean(), 20)           # corners stay black

    def test_warp_quad_round_trip_is_continuous(self) -> None:
        # a gradient must stay a gradient (no flips / transposes)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :] = np.arange(100, dtype=np.uint8)[None, :, None]
        quad = [(0, 0), (99, 0), (99, 99), (0, 99)]
        warped = imageops.warp_quad(image, quad, 100)[..., 0].astype(float)
        self.assertLess(warped[:, 0].mean(), warped[:, -1].mean())
        diffs = np.diff(warped.mean(axis=0))
        self.assertTrue((diffs >= -1e-6).all(), "gradient must be monotone after warping")

    def test_homography_maps_destination_corners(self) -> None:
        quad = [(30.0, 40.0), (250.0, 30.0), (240.0, 260.0), (20.0, 250.0)]
        size = 200
        H = imageops.homography_from_quad(quad, size)
        for (u, v), (x, y) in zip([(0, 0), (size, 0), (size, size), (0, size)], quad):
            denom = H[2, 0] * u + H[2, 1] * v + H[2, 2]
            self.assertAlmostEqual((H[0, 0] * u + H[0, 1] * v + H[0, 2]) / denom, x, places=6)
            self.assertAlmostEqual((H[1, 0] * u + H[1, 1] * v + H[1, 2]) / denom, y, places=6)


class ResizeTest(unittest.TestCase):
    def test_no_upscale(self) -> None:
        image = np.zeros((100, 50, 3), dtype=np.uint8)
        self.assertIs(image, imageops.resize_within(image, 1200))

    def test_aspect_preserved_and_floored(self) -> None:
        image = np.zeros((4032, 3024, 3), dtype=np.uint8)
        resized = imageops.resize_within(image, 1200)
        self.assertEqual((1200, 900), resized.shape[:2])


class ConfigTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        cfg = Config()
        cfg.auto_accept_min = 0.66
        cfg.candidate_deltas = (-0.01, 0.0, 0.01)
        restored = Config.from_dict(cfg.to_dict())
        self.assertEqual(cfg, restored)
        self.assertIsInstance(restored.candidate_deltas, tuple)

    def test_unknown_keys_are_ignored(self) -> None:
        cfg = Config.from_dict({"max_dim": 800, "not_a_field": 3})
        self.assertEqual(800, cfg.max_dim)


if __name__ == "__main__":
    unittest.main()
