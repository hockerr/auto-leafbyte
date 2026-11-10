"""Unit tests for the Lead-owned connected-component labeling (ccl)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from leafbyte import ccl  # noqa: E402
from leafbyte.types import BACKGROUND_LABEL  # noqa: E402


def make_mask(rows: list[str]) -> np.ndarray:
    return np.array([[char == "#" for char in row] for row in rows], dtype=bool)


class LabelComponentsTest(unittest.TestCase):
    def test_single_blob(self) -> None:
        mask = make_mask([
            ".....",
            ".##..",
            ".##..",
            ".....",
        ])
        info = ccl.label_components(mask)
        occupied = info.occupied_labels()
        self.assertEqual(1, len(occupied))
        self.assertEqual(4, info.label_to_size[occupied[0]])
        self.assertEqual((1, 1), info.label_to_member_point[occupied[0]])
        # the whole background is one component, including the outside
        self.assertEqual(mask.size - 4, info.label_to_size[BACKGROUND_LABEL])

    def test_diagonal_is_not_connected(self) -> None:
        mask = make_mask([
            "#..",
            ".#.",
            "..#",
        ])
        info = ccl.label_components(mask)
        self.assertEqual(3, len(info.occupied_labels()))

    def test_enclosed_hole_is_its_own_empty_component(self) -> None:
        mask = make_mask([
            "#####",
            "#...#",
            "#.#.#",
            "#...#",
            "#####",
        ])
        info = ccl.label_components(mask)
        leaf = info.leaf_label()
        self.assertEqual(16, info.label_to_size[leaf])
        # the centre pixel is occupied but isolated -> a second, smaller component
        self.assertEqual([16, 1], [info.label_to_size[l] for l in info.occupied_labels()])
        holes = info.empty_labels()
        self.assertEqual(1, len(holes))  # one 4-connected ring gap around the centre
        self.assertEqual(8, info.label_to_size[holes[0]])
        self.assertIn(leaf, info.empty_to_occupied[holes[0]])

    def test_border_connected_gap_is_background_not_hole(self) -> None:
        mask = make_mask([
            "###..",
            "#....",
            "###..",
        ])
        info = ccl.label_components(mask)
        # the "notch" touches the right border, so it is part of the outside
        self.assertEqual([], info.empty_labels())
        self.assertIn(BACKGROUND_LABEL, info.label_to_size)

    def test_sizes_cover_every_pixel(self) -> None:
        rng = np.random.default_rng(3)
        mask = rng.random((60, 80)) < 0.35
        info = ccl.label_components(mask)
        self.assertEqual(mask.size, sum(info.label_to_size.values()))
        # label map round-trips the mask
        np.testing.assert_array_equal(mask, info.label_map > 0)
        # every label in the map is a canonical representative
        self.assertEqual(set(np.unique(info.label_map).tolist()), set(info.label_to_size))

    def test_empty_to_occupied_is_canonical(self) -> None:
        # two enclosed vertical gaps, framed top and bottom so they are real holes
        mask = make_mask([
            "#######",
            ".#.#.#.",
            ".#.#.#.",
            ".#.#.#.",
            "#######",
        ])
        info = ccl.label_components(mask)
        occupied = set(info.occupied_labels())
        for hole, neighbours in info.empty_to_occupied.items():
            if hole == BACKGROUND_LABEL:
                continue
            self.assertTrue(neighbours.issubset(occupied))
        # both 3 px inner gaps are fully enclosed by the single occupied component
        inner = [h for h in info.empty_labels() if info.label_to_size[h] == 3]
        self.assertEqual(2, len(inner))
        for hole in inner:
            self.assertEqual(occupied, info.empty_to_occupied[hole])

    def test_all_occupied(self) -> None:
        mask = np.ones((5, 5), dtype=bool)
        info = ccl.label_components(mask)
        self.assertEqual(1, len(info.label_to_size))
        self.assertNotIn(BACKGROUND_LABEL, info.label_to_size)

    def test_all_empty(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        info = ccl.label_components(mask)
        self.assertEqual({BACKGROUND_LABEL: 25}, info.label_to_size)
        self.assertEqual((0, 0), info.label_to_member_point[BACKGROUND_LABEL])

    def test_degenerate_shapes(self) -> None:
        self.assertEqual({}, ccl.label_components(np.zeros((0, 0), dtype=bool)).label_to_size)
        single = ccl.label_components(np.ones((1, 1), dtype=bool))
        self.assertEqual((0, 0), single.label_to_member_point[single.leaf_label()])

    def test_multi_component_sizes_ordered(self) -> None:
        mask = make_mask([
            "##........",
            "##..#.....",
            "....###...",
            "..........",
        ])
        info = ccl.label_components(mask)
        sizes = [info.label_to_size[l] for l in info.occupied_labels()]
        self.assertEqual(sorted(sizes, reverse=True), sizes)


if __name__ == "__main__":
    unittest.main()
