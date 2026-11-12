"""Tests for the ecological leaf-trait module (ported from the GUI version)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from leafbyte import traits  # noqa: E402


def rectangle_leaf(width: int = 200, height: int = 50) -> np.ndarray:
    mask = np.zeros((height + 20, width + 20), dtype=bool)
    mask[10:10 + height, 10:10 + width] = True
    return mask


class GeometryTraitTest(unittest.TestCase):
    def test_rectangle_dimensions_and_area(self) -> None:
        mask = rectangle_leaf(200, 50)
        # assume 10 px per cm
        result = traits.compute_traits(mask, None, None, units_per_px=0.1)
        self.assertAlmostEqual(20.0, result["length_cm"], delta=0.6)
        self.assertAlmostEqual(5.0, result["width_cm"], delta=0.6)
        self.assertAlmostEqual(100.0, result["intact_area_cm2"], delta=0.5)
        self.assertAlmostEqual(4.0, result["aspect_ratio"], delta=0.2)
        self.assertAlmostEqual(1.0, result["rectangularity"], delta=0.03)
        # a rectangle fills its convex hull exactly
        self.assertAlmostEqual(1.0, result["solidity"], delta=0.01)

    def test_perimeter_of_rectangle(self) -> None:
        mask = rectangle_leaf(200, 50)
        result = traits.compute_traits(mask, None, None, units_per_px=0.1)
        self.assertAlmostEqual(2 * (20 + 5), result["perimeter_cm"], delta=1.5)

    def test_circle_is_more_circular_than_rectangle(self) -> None:
        yy, xx = np.mgrid[0:300, 0:300]
        circle = (xx - 150) ** 2 + (yy - 150) ** 2 <= 100 ** 2
        circle_traits = traits.compute_traits(circle, None, None, 0.1)
        rect_traits = traits.compute_traits(rectangle_leaf(200, 200), None, None, 0.1)
        self.assertGreater(circle_traits["circularity"], 0.9)
        self.assertLess(rect_traits["circularity"], 0.85)

    def test_holes_reduce_leaf_area_not_intact_area(self) -> None:
        leaf = rectangle_leaf(200, 50)
        hole = np.zeros_like(leaf)
        hole[30:40, 100:120] = True
        result = traits.compute_traits(leaf & ~hole, hole, None, units_per_px=0.1)
        self.assertAlmostEqual(100.0, result["intact_area_cm2"], delta=0.5)
        self.assertAlmostEqual(98.0, result["leaf_area_cm2"], delta=0.6)
        self.assertEqual(1, result["hole_count"])
        self.assertGreater(result["mean_hole_cm2"], 0.0)

    def test_no_scale_gives_dimensionless_only(self) -> None:
        result = traits.compute_traits(rectangle_leaf(200, 50), None, None, None)
        self.assertIsNone(result["intact_area_cm2"])
        self.assertIsNone(result["perimeter_cm"])
        self.assertAlmostEqual(4.0, result["aspect_ratio"], delta=0.2)

    def test_empty_mask_is_all_none(self) -> None:
        result = traits.compute_traits(np.zeros((10, 10), dtype=bool))
        self.assertTrue(all(value is None for value in result.values()))


class ColourTraitTest(unittest.TestCase):
    def test_green_leaf_colour_indices(self) -> None:
        mask = rectangle_leaf(200, 50)
        image = np.zeros((*mask.shape, 3), dtype=np.uint8)
        image[mask] = (60, 120, 50)
        result = traits.compute_traits(mask, None, image, 0.1)
        self.assertAlmostEqual(60, result["mean_r"], delta=1)
        self.assertGreater(result["gcc"], 0.45)
        self.assertGreater(result["exg"], 0.0)


class DryMassTest(unittest.TestCase):
    def test_sla_and_lma(self) -> None:
        record = {"traits": {"leaf_area_cm2": 50.0}}
        traits.attach_dry_mass(record, 250.0)  # 0.25 g for 50 cm²
        self.assertAlmostEqual(200.0, record["sla_cm2_g"], delta=0.01)   # 50 / 0.25
        self.assertAlmostEqual(50.0, record["lma_g_m2"], delta=0.01)     # 0.25 / 0.005 m²

    def test_missing_mass_leaves_none(self) -> None:
        record = {"traits": {"leaf_area_cm2": 50.0}}
        traits.attach_dry_mass(record, None)
        self.assertIsNone(record["sla_cm2_g"])

    def test_columns_are_exported(self) -> None:
        self.assertIn("perimeter_cm", traits.TRAIT_COLUMNS)
        self.assertIn("solidity", traits.TRAIT_COLUMNS)
        self.assertEqual(["dry_mass_mg", "sla_cm2_g", "lma_g_m2"], traits.MASS_COLUMNS)


if __name__ == "__main__":
    unittest.main()
