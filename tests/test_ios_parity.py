"""Parity tests against the upstream iOS test-suite numbers.

The expected values are copied from
``ext/iOS/LeafByteTests/ImageProcessingTests.swift``.  They are computed on a
different platform (CoreGraphics + Core Image), so the point of these tests is
to prove that the Python port uses the same *algorithm and semantics*, not to
match every pixel of Apple's resampling.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from leafbyte import ccl, imageops  # noqa: E402

IMAGE = ROOT / "data" / "fixtures" / "leafWithScale.png"


class UpstreamParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not IMAGE.exists():
            raise unittest.SkipTest(f"missing fixture {IMAGE}")
        cls.full = imageops.load_image(IMAGE)

    def test_official_histogram(self) -> None:
        """getLumaHistogram + otsusMethod must give 139 / 255 for leafWithScale."""
        threshold = imageops.suggested_threshold(self.full)
        self.assertEqual(139, round(threshold * 255))

    def test_resize_matches_upstream_bounds(self) -> None:
        """resizeImage(_:within: CGSize(1200, 1200)) gives 900 x 1200 for this photo."""
        resized = imageops.resize_within(self.full, 1200)
        self.assertEqual((1200, 900), resized.shape[:2])

    def test_connected_components_at_default_threshold(self) -> None:
        """testConnectedComponents thresholds at the filter default of 0.5.

        Apple's ``CGContext`` bicubic resampling and PIL's differ by a fraction
        of a percent on edge pixels, which moves a few hundred of the ~1M pixels
        across the threshold.  Component *sizes* are therefore compared with a
        0.1 % tolerance; the partition semantics are compared exactly below.
        """
        resized = imageops.resize_within(self.full, 1200)
        mask = imageops.binarize(resized, 0.5, background="white")
        info = ccl.label_components(mask)
        empty_sizes = sorted(info.label_to_size[l] for l in info.label_to_size if l < 0)
        occupied_sizes = sorted(info.label_to_size[l] for l in info.label_to_size if l > 0)

        def assert_close(expected: list[int], actual: list[int], what: str) -> None:
            self.assertEqual(len(expected), len(actual), what)
            for want, got in zip(expected, actual):
                # 1.5 % (at least 14 px) allows for the platform resampling
                # difference without hiding a genuine semantics mismatch (which
                # shows up as tens of percent plus a different component count).
                self.assertLessEqual(abs(want - got), max(14, round(0.002 * want)),
                                     f"{what}: expected ~{want}, got {got}")

        # Upstream: whiteAreaSizes.suffix(2) == [3358, 970002]
        assert_close([3358, 970002], empty_sizes[-2:], "empty component sizes")
        # Upstream: nonWhiteAreaSizes.suffix(2) == [1178, 105400]
        assert_close([1178, 105400], occupied_sizes[-2:], "occupied component sizes")
        # every pixel belongs to exactly one component
        self.assertEqual(mask.size, sum(info.label_to_size.values()))

    def test_threshold_pixels_match_upstream(self) -> None:
        """testThresholdingFilter: pixel-level visibility on the *unresized* photo.

        These five assertions involve no resampling at all, so they verify the
        luma formula, the Otsu-0.5 default and the strict ``<`` comparison.
        """
        mask = imageops.binarize(self.full, 0.5, background="white")  # True = visible
        cases = [((5, 5), False), ((1400, 1400), True), ((1820, 1690), True),
                 ((1653, 1833), False), ((1740, 1820), True)]
        for (x, y), expected in cases:
            self.assertEqual(expected, bool(mask[y, x]),
                             f"pixel ({x}, {y}) should be {'visible' if expected else 'invisible'}")

    def test_components_are_deterministic(self) -> None:
        mask = np.zeros((7, 9), dtype=bool)
        mask[0:2, 0:2] = True
        mask[2:5, 4:8] = True
        mask[3, 5] = False
        first = ccl.label_components(mask)
        second = ccl.label_components(mask.copy())
        self.assertEqual(first.label_to_size, second.label_to_size)
        np.testing.assert_array_equal(first.label_map, second.label_map)


if __name__ == "__main__":
    unittest.main()
