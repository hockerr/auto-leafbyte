"""Unit tests for :mod:`leafbyte.scale` (scale-mark detection + rectification).

Owner: ``scale``.  Run with::

    PYTHONPATH=src python -m unittest discover -s tests -t . -v

Covers
- pure geometry: 4-point ordering, quad metrics, homography solve, rejection of
  degenerate quadrilaterals;
- candidate filtering from components (leaf / frame strip / dust must not be
  treated as scale marks);
- the full ``detect`` path on synthetic photos (rotated, perspective-warped,
  noisy, faded dot, missing dot, dust specks, degenerate layouts);
- rectification (output size, projective correctness, ``scale_px_length``);
- determinism, runtime budget and a smoke test on the real benchmark image.
"""

from __future__ import annotations

import math
import time
import unittest
from pathlib import Path

import numpy as np

from leafbyte import ccl, imageops, scale
from leafbyte.config import Config
from leafbyte.types import ScaleDetection

ROOT = Path(__file__).resolve().parents[1]
CO1 = ROOT / "data" / "paper" / "images" / "CO1.png"
AI2 = ROOT / "data" / "paper" / "images" / "AI2.png"

if not CO1.exists():  # the upstream photos are fetched, not committed
    raise unittest.SkipTest("paper images missing - run tools/fetch_paper_data.py")

DEFAULT_HEIGHT = 1200
DEFAULT_WIDTH = 900
#: corner dots of the "standard" synthetic photo, ordered TL, TR, BR, BL
BASE_QUAD = [(150.0, 300.0), (750.0, 280.0), (770.0, 900.0), (140.0, 920.0)]
#: a synthetic trapezoid (perspective-looking) mark layout
TRAPEZOID_QUAD = [(150.0, 300.0), (760.0, 250.0), (820.0, 950.0), (100.0, 1000.0)]


# ---------------------------------------------------------------------------
# synthetic image helpers
# ---------------------------------------------------------------------------
def make_photo(
    dots=() ,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    dot_radius: int = 11,
    dot_value: int = 25,
    specks=(),
    speck_radius: int = 3,
    marker=None,
    marker_radius: int = 6,
    noise: int = 0,
    seed: int = 1234,
    background: int = 245,
):
    """White canvas + black discs (deterministic, numpy only)."""
    image = np.full((height, width, 3), background, dtype=np.uint8)
    ys, xs = np.ogrid[:height, :width]
    for cx, cy in dots:
        image[(xs - cx) ** 2 + (ys - cy) ** 2 <= dot_radius * dot_radius] = dot_value
    for cx, cy in specks:
        image[(xs - cx) ** 2 + (ys - cy) ** 2 <= speck_radius * speck_radius] = 30
    if marker is not None:
        mx, my = marker
        disc = (xs - mx) ** 2 + (ys - my) ** 2 <= marker_radius * marker_radius
        image[disc] = (220, 20, 20)
    if noise:
        generator = np.random.default_rng(seed)
        deltas = generator.integers(-noise, noise + 1, size=image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + deltas, 0, 255).astype(np.uint8)
    return image


def points_close(actual, expected, tolerance: float = 2.0) -> bool:
    got = np.asarray(actual, dtype=np.float64).reshape(-1, 2)
    want = np.asarray(expected, dtype=np.float64).reshape(-1, 2)
    return got.shape == want.shape and bool(np.all(np.linalg.norm(got - want, axis=1) <= tolerance))


def rotate(points, degrees: float, centre=(450.0, 600.0)):
    angle = math.radians(degrees)
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    centre_array = np.asarray(centre, dtype=np.float64)
    return [tuple(centre_array + rotation @ (np.asarray(p, dtype=np.float64) - centre_array))
            for p in points]


# ---------------------------------------------------------------------------
# pure geometry
# ---------------------------------------------------------------------------
class TestQuadOrdering(unittest.TestCase):
    def test_axis_aligned_from_any_order(self):
        square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        for permutation in ([1, 0, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1], [0, 2, 1, 3]):
            shuffled = [square[i] for i in permutation]
            ordered = scale.order_quad(shuffled)
            np.testing.assert_allclose(ordered, np.asarray(square), atol=1e-9)

    def test_rotated_square(self):
        rotated = rotate([(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)], 25.0,
                         centre=(100.0, 100.0))
        ordered = scale.order_quad(rotated)
        # TL has the smallest x+y, then clockwise: TR, BR, BL
        sums = ordered[:, 0] + ordered[:, 1]
        self.assertEqual(int(np.argmin(sums)), 0)
        np.testing.assert_allclose(ordered[0], rotated[0], atol=1e-9)
        np.testing.assert_allclose(ordered[1], rotated[1], atol=1e-9)
        np.testing.assert_allclose(ordered[2], rotated[2], atol=1e-9)
        np.testing.assert_allclose(ordered[3], rotated[3], atol=1e-9)

    def test_perspective_quad(self):
        quad = np.asarray(TRAPEZOID_QUAD)
        ordered = scale.order_quad(quad[[2, 0, 3, 1]])
        np.testing.assert_allclose(ordered, quad, atol=1e-9)

    def test_wrong_number_of_points(self):
        with self.assertRaises(ValueError):
            scale.order_quad([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])


class TestQuadMetrics(unittest.TestCase):
    def test_square(self):
        metrics = scale.quad_metrics([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)])
        np.testing.assert_allclose(metrics["sides"], [100.0] * 4, atol=1e-9)
        self.assertAlmostEqual(metrics["side_ratio"], 1.0)
        self.assertAlmostEqual(metrics["diagonal_ratio"], 1.0)
        self.assertAlmostEqual(metrics["area"], 10000.0, places=6)
        self.assertTrue(metrics["convex"])
        np.testing.assert_allclose(metrics["angles"], [90.0] * 4, atol=1e-6)
        self.assertGreater(metrics["squareness"], 0.99)

    def test_perspective_rectangle_is_not_a_square(self):
        metrics = scale.quad_metrics([(0.0, 0.0), (400.0, 0.0), (400.0, 100.0), (0.0, 100.0)])
        self.assertLess(metrics["side_ratio"], 0.3)
        self.assertLess(metrics["squareness"], 0.7)


class TestHomography(unittest.TestCase):
    def test_roundtrip_known_projectivity(self):
        src = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        known = np.array([[2.0, 0.3, 10.0],
                          [0.1, 1.5, 20.0],
                          [5e-4, 2e-4, 1.0]])
        dst = scale.apply_homography(known, src)
        solved = scale.solve_homography(src, dst)
        np.testing.assert_allclose(solved, known / known[2, 2], atol=1e-8)
        probes = np.random.default_rng(0).random((12, 2))
        np.testing.assert_allclose(scale.apply_homography(solved, probes),
                                   scale.apply_homography(known, probes), atol=1e-7)

    def test_quad_to_square(self):
        quad = np.asarray(TRAPEZOID_QUAD)
        size = 500
        dst = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
        homography = scale.solve_homography(quad, dst)
        np.testing.assert_allclose(scale.apply_homography(homography, quad),
                                   np.asarray(dst, dtype=np.float64), atol=1e-6)

    def test_degenerate_points_raise(self):
        collinear = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
        with self.assertRaises(ValueError):
            scale.solve_homography(collinear, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        duplicated = [(0.0, 0.0), (0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        with self.assertRaises(ValueError):
            scale.solve_homography(duplicated, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    def test_too_few_points_raise(self):
        with self.assertRaises(ValueError):
            scale.solve_homography([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
                                   [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])


class TestQuadRejection(unittest.TestCase):
    """The scorers must reject garbage quads instead of returning them."""

    @staticmethod
    def _evaluate(quads, sizes=None, shapes=None, all_sizes=None, all_shapes=None,
                  width=1000, height=1000):
        quads = np.asarray(quads, dtype=np.float64).reshape(-1, 4, 2)
        count = quads.shape[0]
        sizes = (np.full((count, 4), 400.0) if sizes is None
                 else np.asarray(sizes, dtype=np.float64).reshape(count, 4))
        shapes = (np.ones((count, 4)) if shapes is None
                  else np.asarray(shapes, dtype=np.float64).reshape(count, 4))
        combos = None
        if all_sizes is not None:
            combos = np.array([[0, 1, 2, 3]], dtype=np.intp)[:count]
            if count > 1:  # pragma: no cover - only used for smoke checks
                raise AssertionError("all_sizes variant supports a single quad")
        return scale._evaluate_quads(quads, sizes, shapes, width, height,
                                     all_sizes=all_sizes, all_shapes=all_shapes,
                                     combos=combos)

    def test_convex_square_is_valid(self):
        result = self._evaluate([[100.0, 100.0], [900.0, 100.0], [900.0, 900.0], [100.0, 900.0]])
        self.assertTrue(bool(result["valid"][0]))
        self.assertGreater(float(result["quality"][0]), 0.85)

    def test_bowtie_is_rejected(self):
        result = self._evaluate([[100.0, 100.0], [900.0, 900.0], [900.0, 100.0], [100.0, 900.0]])
        self.assertFalse(bool(result["valid"][0]))

    def test_tiny_quad_is_rejected(self):
        result = self._evaluate([[480.0, 480.0], [520.0, 480.0], [520.0, 520.0], [480.0, 520.0]])
        self.assertFalse(bool(result["valid"][0]))

    def test_huge_quad_is_rejected(self):
        result = self._evaluate([[0.0, 0.0], [999.0, 0.0], [999.0, 999.0], [0.0, 999.0]])
        self.assertFalse(bool(result["valid"][0]))

    def test_elongated_quad_is_rejected(self):
        result = self._evaluate([[100.0, 450.0], [900.0, 450.0], [900.0, 550.0], [100.0, 550.0]])
        self.assertFalse(bool(result["valid"][0]))

    def test_heterogeneous_mark_sizes_are_rejected(self):
        quad = [[100.0, 100.0], [900.0, 100.0], [900.0, 900.0], [100.0, 900.0]]
        result = self._evaluate([quad], sizes=[[400.0, 410.0, 390.0, 20.0]])
        self.assertFalse(bool(result["valid"][0]))

    def test_quad_of_dust_specks_is_deprioritised_by_prominence(self):
        dust_quad = [[200.0, 200.0], [800.0, 200.0], [800.0, 800.0], [200.0, 800.0]]
        # without any bigger blob around, the geometry alone is acceptable
        neutral = self._evaluate([dust_quad], sizes=[[100.0] * 4])
        self.assertTrue(bool(neutral["valid"][0]))
        # but if big dot-like blobs exist elsewhere, the dust quad must lose
        prominent = self._evaluate(
            [dust_quad], sizes=[[100.0] * 4],
            all_sizes=np.array([100.0, 101.0, 102.0, 480.0, 500.0, 510.0, 520.0]),
            all_shapes=np.ones(7))
        self.assertLess(float(prominent["quality"][0]), 0.75 * float(neutral["quality"][0]))


# ---------------------------------------------------------------------------
# candidate filtering from components
# ---------------------------------------------------------------------------
class TestCandidateFilter(unittest.TestCase):
    def test_leaf_frame_and_dust_are_not_marks(self):
        height, width = 1200, 900
        mask = np.zeros((height, width), dtype=bool)
        ys, xs = np.ogrid[:height, :width]
        # the four real marks
        for cx, cy in BASE_QUAD:
            mask[(xs - cx) ** 2 + (ys - cy) ** 2 <= 11 * 11] = True
        # a big leaf blob in the middle (~3% of the image -> filtered by area)
        mask[((xs - 450) / 220.0) ** 2 + ((ys - 600) / 300.0) ** 2 <= 1.0] = True
        # a dark frame strip glued to the top border: long and thin
        mask[0:12, :] = True
        # dust specks
        for cx, cy in [(700, 100), (200, 1100), (820, 700)]:
            mask[cy - 2:cy + 3, cx - 2:cx + 3] = True

        components = ccl.label_components(mask)
        candidates = scale._candidates_from_components(components, Config(), relaxed=False)
        # The four real marks must survive, and be ranked first by dot-likeness.
        self.assertGreaterEqual(len(candidates), 4, msg=f"candidates={candidates}")
        top = candidates[:4]
        top_centroids = sorted((round(c["cx"], 1), round(c["cy"], 1)) for c in top)
        want = sorted((round(x, 1), round(y, 1)) for x, y in BASE_QUAD)
        for got_point, want_point in zip(top_centroids, want):
            self.assertLess(math.dist(got_point, want_point), 0.5)
        # ... and nothing leaf-sized / frame-strip-like may be a candidate.
        for candidate in candidates:
            self.assertLess(candidate["size"], 1000)   # leaf blob is much bigger
            self.assertGreaterEqual(candidate["aspect"], 1.0)
            self.assertLess(candidate["aspect"], 1.2)  # strips are elongated
            self.assertFalse(candidate["border_touch"])
            self.assertLess(candidate["area_fraction"], 0.001)

    def test_border_touching_mark_is_dropped(self):
        height, width = 200, 200
        mask = np.zeros((height, width), dtype=bool)
        mask[0:20, 90:110] = True  # dot cut by the top border
        components = ccl.label_components(mask)
        candidates = scale._candidates_from_components(components, Config(), relaxed=False)
        self.assertEqual(candidates, [])


# ---------------------------------------------------------------------------
# full detection path on synthetic photos
# ---------------------------------------------------------------------------
class TestDetectSynthetic(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_four_dots(self):
        detection = scale.detect(make_photo(BASE_QUAD), self.cfg)
        self.assertTrue(detection.found)
        self.assertEqual(detection.method, "quad")
        self.assertGreater(detection.quality, 0.85)
        self.assertTrue(points_close(detection.points, BASE_QUAD, tolerance=2.0))
        self.assertEqual(len(detection.candidates), 4)
        self.assertEqual(sum(1 for c in detection.candidates if c["selected"]), 4)
        self.assertEqual(detection.details["side_min_px"] > 0, True)

    def test_rotated_dots(self):
        rotated = rotate(BASE_QUAD, 15.0)
        detection = scale.detect(make_photo(rotated), self.cfg)
        self.assertTrue(detection.found)
        self.assertTrue(points_close(detection.points, rotated, tolerance=2.0))

    def test_perspective_quad(self):
        detection = scale.detect(make_photo(TRAPEZOID_QUAD), self.cfg)
        self.assertTrue(detection.found)
        self.assertGreater(detection.quality, 0.8)
        self.assertTrue(points_close(detection.points, TRAPEZOID_QUAD, tolerance=2.0))

    def test_noisy_photo(self):
        detection = scale.detect(make_photo(BASE_QUAD, noise=18), self.cfg)
        self.assertTrue(detection.found)
        self.assertTrue(points_close(detection.points, BASE_QUAD, tolerance=2.0))

    def test_dust_specks_do_not_win(self):
        specks = [(80.0, 120.0), (700.0, 90.0), (860.0, 500.0),
                  (300.0, 1080.0), (640.0, 760.0), (90.0, 760.0)]
        detection = scale.detect(make_photo(BASE_QUAD, specks=specks), self.cfg)
        self.assertTrue(detection.found)
        self.assertTrue(points_close(detection.points, BASE_QUAD, tolerance=2.0))

    def test_faded_dot_still_detected(self):
        # one weak (small, low-contrast) dot: still a legitimate mark
        image = make_photo(BASE_QUAD[:3])
        ys, xs = np.ogrid[:DEFAULT_HEIGHT, :DEFAULT_WIDTH]
        cx, cy = BASE_QUAD[3]
        image[(xs - cx) ** 2 + (ys - cy) ** 2 <= 6 * 6] = 120
        detection = scale.detect(image, self.cfg)
        self.assertTrue(detection.found)
        self.assertIn("uneven scale-mark sizes", detection.flags)

    def test_three_dots_infer_fourth_corner(self):
        # three corners of a perfect rectangle: parallelogram completion is exact
        rectangle = [(200.0, 300.0), (700.0, 300.0), (700.0, 900.0), (200.0, 900.0)]
        detection = scale.detect(make_photo(rectangle[:3]), self.cfg)
        self.assertTrue(detection.found)
        self.assertEqual(detection.method, "quad_3dot")
        self.assertIn("4th scale mark inferred", detection.flags)
        self.assertLess(detection.quality, 0.75)  # low confidence by construction
        self.assertTrue(points_close(detection.points, rectangle, tolerance=3.0))
        self.assertEqual(len(detection.points), 4)

    def test_three_dots_beat_dust_quad(self):
        rectangle = [(200.0, 300.0), (700.0, 300.0), (700.0, 900.0), (200.0, 900.0)]
        specks = [(120.0, 120.0), (760.0, 140.0), (800.0, 700.0),
                  (320.0, 1060.0), (600.0, 180.0), (420.0, 760.0)]
        detection = scale.detect(make_photo(rectangle[:3], specks=specks), self.cfg)
        self.assertTrue(detection.found)
        self.assertEqual(detection.method, "quad_3dot")
        self.assertTrue(points_close(detection.points, rectangle, tolerance=4.0))

    def test_two_dots_fail_cleanly(self):
        detection = scale.detect(make_photo(BASE_QUAD[:2]), self.cfg)
        self.assertFalse(detection.found)
        self.assertEqual(detection.method, "none")
        self.assertEqual(detection.quality, 0.0)
        self.assertEqual(len(detection.points), 0)
        self.assertTrue(any("candidate" in flag for flag in detection.flags))

    def test_elongated_layout_fails_cleanly(self):
        rectangle_4x1 = [(100.0, 500.0), (800.0, 500.0), (800.0, 700.0), (100.0, 700.0)]
        detection = scale.detect(make_photo(rectangle_4x1), self.cfg)
        self.assertFalse(detection.found)
        self.assertTrue(any("near-square" in flag for flag in detection.flags))

    def test_blank_canvas_fails(self):
        detection = scale.detect(np.full((1200, 900, 3), 245, dtype=np.uint8), self.cfg)
        self.assertFalse(detection.found)
        self.assertEqual(detection.method, "none")

    def test_dust_specks_only_fail_cleanly(self):
        # Four similar dust specs can geometrically form a rough square; the
        # mark-area floor must stop them from being reported as scale marks.
        specks = [(80.0, 120.0), (700.0, 90.0), (800.0, 700.0), (320.0, 1060.0)]
        detection = scale.detect(make_photo(specks=specks), self.cfg)
        self.assertFalse(detection.found)
        self.assertEqual(detection.method, "none")

    def test_deterministic(self):
        image = make_photo(BASE_QUAD, specks=[(50.0, 60.0), (800.0, 1100.0)])
        first = scale.detect(image, self.cfg)
        second = scale.detect(image, self.cfg)
        self.assertEqual(first.found, second.found)
        self.assertEqual(first.method, second.method)
        self.assertEqual(first.quality, second.quality)
        self.assertEqual([tuple(p) for p in first.points], [tuple(p) for p in second.points])

    def test_runtime_budget(self):
        image = make_photo(BASE_QUAD, noise=10)
        start = time.perf_counter()
        scale.detect(image, self.cfg)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 3.0, msg=f"detect took {elapsed:.3f}s on a 900x1200 image")


# ---------------------------------------------------------------------------
# rectification
# ---------------------------------------------------------------------------
class TestRectify(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_size_and_scale_length(self):
        detection = scale.detect(make_photo(BASE_QUAD), self.cfg)
        self.assertTrue(detection.found)
        rectified = scale.rectify(make_photo(BASE_QUAD), detection, self.cfg)
        self.assertIsNotNone(rectified)
        bbox_w = max(p[0] for p in detection.points) - min(p[0] for p in detection.points)
        bbox_h = max(p[1] for p in detection.points) - min(p[1] for p in detection.points)
        expected = int(min(self.cfg.max_dim, math.floor(min(bbox_w, bbox_h))))
        self.assertEqual(rectified.size, expected)
        self.assertEqual(rectified.scale_px_length, float(expected))
        self.assertEqual(rectified.image.shape, (expected, expected, 3))
        self.assertEqual(rectified.image.dtype, np.uint8)

    def test_homography_maps_quad_to_square(self):
        image = make_photo(BASE_QUAD)
        detection = scale.detect(image, self.cfg)
        rectified = scale.rectify(image, detection, self.cfg)
        size = rectified.size
        corners = np.asarray(scale.order_quad(detection.points))
        expected = np.array([[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]])
        mapped = scale.apply_homography(rectified.homography, corners)
        np.testing.assert_allclose(mapped, expected, atol=1e-4 * size)

    def test_marker_lands_in_expected_quadrant(self):
        marker_quad = [(200.0, 300.0), (700.0, 300.0), (700.0, 900.0), (200.0, 900.0)]
        # marker at (25%, 25%) of the square in quad coordinates
        marker = (200.0 + 0.25 * 500.0, 300.0 + 0.25 * 600.0)
        image = make_photo(marker_quad, marker=marker, marker_radius=7)
        detection = scale.detect(image, self.cfg)
        rectified = scale.rectify(image, detection, self.cfg)
        size = rectified.size
        out = rectified.image.astype(np.int16)
        red = (out[:, :, 0] > 150) & (out[:, :, 1] < 100) & (out[:, :, 2] < 100)
        self.assertTrue(red.any(), "rectified image lost the red marker")
        ys, xs = np.nonzero(red)
        cx, cy = float(xs.mean()), float(ys.mean())
        self.assertAlmostEqual(cx / size, 0.25, delta=0.02)
        self.assertAlmostEqual(cy / size, 0.25, delta=0.02)

    def test_marker_position_under_perspective(self):
        quad = np.asarray(TRAPEZOID_QUAD, dtype=np.float64)
        unit = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        project = scale.solve_homography(unit, quad)
        marker = scale.apply_homography(project, [(0.3, 0.6)])[0]
        image = make_photo(quad, marker=tuple(marker), marker_radius=7)
        detection = scale.detect(image, self.cfg)
        self.assertTrue(detection.found)
        rectified = scale.rectify(image, detection, self.cfg)
        size = rectified.size
        out = rectified.image.astype(np.int16)
        red = (out[:, :, 0] > 150) & (out[:, :, 1] < 100) & (out[:, :, 2] < 100)
        self.assertTrue(red.any(), "rectified image lost the red marker")
        ys, xs = np.nonzero(red)
        self.assertAlmostEqual(float(xs.mean()) / size, 0.3, delta=0.03)
        self.assertAlmostEqual(float(ys.mean()) / size, 0.6, delta=0.03)

    def test_rectify_without_detection_returns_none(self):
        self.assertIsNone(scale.rectify(make_photo(BASE_QUAD), None, self.cfg))
        self.assertIsNone(scale.rectify(make_photo(BASE_QUAD),
                                        ScaleDetection(found=False), self.cfg))

    def test_rectify_degenerate_points_returns_none(self):
        detection = ScaleDetection(found=True,
                                   points=[(10.0, 10.0), (20.0, 20.0), (30.0, 30.0), (40.0, 40.0)],
                                   quality=0.9, method="manual")
        self.assertIsNone(scale.rectify(make_photo(BASE_QUAD), detection, self.cfg))

    def test_detect_and_rectify(self):
        detection, rectified = scale.detect_and_rectify(make_photo(BASE_QUAD), self.cfg)
        self.assertTrue(detection.found)
        self.assertIsNotNone(rectified)
        self.assertEqual(rectified.image.shape[0], rectified.image.shape[1])


# ---------------------------------------------------------------------------
# real benchmark images (skipped when the data directory is absent)
# ---------------------------------------------------------------------------
class TestPaperImages(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def _normalized(self, path):
        image = imageops.load_image(path)
        return imageops.resize_within(image, self.cfg.max_dim)

    @unittest.skipUnless(CO1.exists(), "benchmark images not available")
    def test_co1(self):
        normalized = self._normalized(CO1)
        self.assertEqual(normalized.shape[:2], (1200, 900))
        detection = scale.detect(normalized, self.cfg)
        self.assertTrue(detection.found)
        self.assertEqual(detection.method, "quad")
        self.assertGreater(detection.quality, 0.9)
        # visually confirmed dot centroids in CO1.png (ordered TL, TR, BR, BL)
        expected = [(75.9, 227.3), (846.9, 216.3), (855.9, 992.4), (83.9, 996.4)]
        self.assertTrue(points_close(detection.points, expected, tolerance=5.0))
        rectified = scale.rectify(normalized, detection, self.cfg)
        self.assertIsNotNone(rectified)
        self.assertEqual(rectified.scale_px_length, float(rectified.size))
        self.assertGreaterEqual(rectified.size, 700)
        self.assertLessEqual(rectified.size, 820)

    @unittest.skipUnless(AI2.exists(), "benchmark images not available")
    def test_ai2_faded_dot(self):
        normalized = self._normalized(AI2)
        detection = scale.detect(normalized, self.cfg)
        self.assertTrue(detection.found)
        self.assertEqual(detection.method, "quad")
        # AI2 has one genuinely faded bottom-left dot: surface it, don't hide it
        self.assertIn("uneven scale-mark sizes", detection.flags)
        self.assertLess(detection.quality, 0.97)
        self.assertGreater(detection.quality, 0.7)
        rectified = scale.rectify(normalized, detection, self.cfg)
        self.assertIsNotNone(rectified)


if __name__ == "__main__":
    unittest.main()
