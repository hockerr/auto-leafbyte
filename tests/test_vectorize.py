"""Tests for sub-pixel contour tracing and SVG export (vectorize)."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from leafbyte import vectorize  # noqa: E402


def circle_field(size: int = 200, radius: float = 60.0, centre=None) -> np.ndarray:
    """A smooth field that is positive inside a disc (like ``ExG - cut``)."""
    centre = centre or (size / 2.0, size / 2.0)
    yy, xx = np.mgrid[0:size, 0:size]
    distance = np.hypot(xx - centre[0], yy - centre[1])
    return radius - distance  # > 0 inside the disc


class MarchingSquaresTest(unittest.TestCase):
    def test_single_disc_gives_one_closed_contour(self) -> None:
        polygons = vectorize.marching_squares(circle_field(), 0.0)
        self.assertEqual(1, len(polygons))
        polygon = polygons[0]
        self.assertGreater(len(polygon), 100)
        area = abs(vectorize.polygon_area(polygon))
        expected = np.pi * 60.0 ** 2
        self.assertLess(abs(area - expected) / expected, 0.02,
                        f"area {area:.1f} vs {expected:.1f}")

    def test_contour_is_sub_pixel(self) -> None:
        """Vertices must sit between pixels, not on the integer grid."""
        polygon = vectorize.marching_squares(circle_field(), 0.0)[0]
        fractional = sum(1 for x, y in polygon
                         if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6)
        self.assertGreater(fractional / len(polygon), 0.8)

    def test_two_discs_give_two_contours(self) -> None:
        field = np.maximum(circle_field(200, 40, (60, 60)), circle_field(200, 40, (140, 140)))
        polygons = vectorize.marching_squares(field, 0.0)
        self.assertEqual(2, len(polygons))

    def test_empty_field(self) -> None:
        self.assertEqual([], vectorize.marching_squares(np.full((20, 20), -1.0), 0.0))
        self.assertEqual([], vectorize.marching_squares(np.full((20, 20), 1.0), 0.0))


class SimplifyTest(unittest.TestCase):
    def test_simplify_keeps_shape(self) -> None:
        polygon = vectorize.marching_squares(circle_field(), 0.0)[0]
        simplified = vectorize.simplify(polygon, tolerance=0.5)
        self.assertLess(len(simplified), len(polygon))
        self.assertGreater(len(simplified), 20)
        original_area = abs(vectorize.polygon_area(polygon))
        simplified_area = abs(vectorize.polygon_area(simplified))
        self.assertLess(abs(simplified_area - original_area) / original_area, 0.01)

    def test_simplify_is_identity_for_tiny_input(self) -> None:
        triangle = [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]
        self.assertEqual(triangle, vectorize.simplify(triangle, 1.0))


class SVGTest(unittest.TestCase):
    def test_svg_has_leaf_and_holes_with_real_units(self) -> None:
        field = circle_field(200, 60)
        hole = circle_field(200, 12, (100, 100))
        contours = vectorize.marching_squares(np.where(hole > 0, -hole, field), 0.0)
        self.assertGreaterEqual(len(contours), 2)

        svg = vectorize.contour_svg(contours, size=200, scale_cm=17.0, title="demo")
        self.assertIn("<svg", svg)
        self.assertIn("width=\"17.0000cm\"", svg)
        paths = re.findall(r'<path d="([^"]+)"', svg)
        self.assertGreaterEqual(len(paths), 2)  # leaf + hole (each also in the outline group)

        match = re.search(r'data-leaf-area-cm2="([\d.]+)" data-hole-area-cm2="([\d.]+)"', svg)
        self.assertIsNotNone(match)
        leaf_cm2, hole_cm2 = float(match.group(1)), float(match.group(2))
        # 200 px == 17 cm  ->  1 px == 0.085 cm
        expected_leaf = np.pi * 60.0 ** 2 * (17.0 / 200.0) ** 2
        # marching squares puts vertices on the 0.5-level crossing (half a pixel
        # outside the pixel-centre polygon), so allow a few percent here; the
        # reported area uses the pixel count, not the polygon.
        self.assertLess(abs(leaf_cm2 - expected_leaf) / expected_leaf, 0.05)
        self.assertGreater(hole_cm2, 0.0)

    def test_svg_drops_spurious_specks(self) -> None:
        field = circle_field(200, 60)
        # add a 1 px fake blob far from the disc
        field[5, 5] = 1.0
        contours = vectorize.marching_squares(field, 0.0)
        svg = vectorize.contour_svg(contours, size=200, scale_cm=17.0)
        self.assertIn('data-dropped-specks="1"', svg)

    def test_empty_contours_produce_valid_svg(self) -> None:
        svg = vectorize.contour_svg([], size=100, scale_cm=17.0)
        self.assertIn("<svg", svg)


class AnnotationTest(unittest.TestCase):
    """Scale bar + sample label must be present in every export format."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.contours = vectorize.marching_squares(circle_field(200, 60), 0.0)

    def test_svg_has_scale_bar_and_label(self) -> None:
        svg = vectorize.contour_svg(self.contours, size=200, scale_cm=17.0,
                                    label="140盐肤木-1", scale_bar_cm=5.0)
        self.assertIn("annotations", svg)
        self.assertIn("140盐肤木-1", svg)
        self.assertIn("<rect", svg)                       # the bar itself
        self.assertIn(">cm<", svg)
        # canvas grew by the annotation band: 17 cm wide -> 17 x (17+2.4) cm
        self.assertIn('width="17.0000cm"', svg)
        self.assertIn('height="19.4000cm"', svg)

    def test_svg_can_disable_annotations(self) -> None:
        svg = vectorize.contour_svg(self.contours, size=200, scale_cm=17.0,
                                    label="x", annotate=False)
        self.assertNotIn("annotations", svg)
        self.assertIn('height="17.0000cm"', svg)

    def test_pdf_is_structurally_valid(self) -> None:
        data = vectorize.contour_pdf(self.contours, size=200, scale_cm=17.0,
                                     label="140盐肤木-1")
        self.assertTrue(data.startswith(b"%PDF-1.4"))
        self.assertTrue(data.rstrip().endswith(b"%%EOF"))
        self.assertIn(b"/MediaBox", data)
        self.assertIn(b"/Subtype /Image", data)           # CJK label bitmap
        self.assertIn(b"Tj", data)                        # "0", "5", "cm"
        match = re.search(rb"startxref\s+(\d+)", data)
        self.assertIsNotNone(match)
        start = int(match.group(1))
        self.assertEqual(b"xref", data[start:start + 4])
        offsets = re.findall(rb"(\d{10}) 00000 n", data[start:])
        self.assertGreaterEqual(len(offsets), 5)
        for index, offset in enumerate(offsets, start=1):
            self.assertTrue(data[int(offset):int(offset) + 20].startswith(f"{index} 0 obj".encode()),
                            f"xref offset for object {index} is wrong")

    def test_pdf_cross_references_point_at_the_right_objects(self) -> None:
        """Regression: an off-by-one used to point /F1 at the content stream."""
        data = vectorize.contour_pdf(self.contours, size=200, scale_cm=17.0,
                                     label="140盐肤木-1")
        self.assertEqual([], vectorize.verify_pdf_bytes(data))
        # the page resources must reference a Type1 font and an image XObject
        font_ref = int(re.search(rb"/Font << /F1 (\d+) 0 R >>", data).group(1))
        self.assertIn(b"/Subtype /Type1", data.split(f"{font_ref} 0 obj".encode())[1][:200])
        image_ref = int(re.search(rb"/Im1 (\d+) 0 R", data).group(1))
        self.assertIn(b"/Subtype /Image", data.split(f"{image_ref} 0 obj".encode())[1][:200])

    def test_pdf_page_size_is_physical(self) -> None:
        data = vectorize.contour_pdf(self.contours, size=200, scale_cm=17.0, label="")
        match = re.search(rb"/MediaBox \[0 0 ([\d.]+) ([\d.]+)\]", data)
        self.assertIsNotNone(match)
        width_pt, height_pt = float(match.group(1)), float(match.group(2))
        self.assertAlmostEqual(17.0 * 72 / 2.54, width_pt, places=2)
        self.assertAlmostEqual(19.4 * 72 / 2.54, height_pt, places=2)

    def test_plate_png(self) -> None:
        native = vectorize.contour_plate_png(self.contours, size=200, scale_cm=17.0,
                                             label="140盐肤木-1", max_width_px=400)
        self.assertEqual(200, native.width, "small plates are not upscaled")
        plate = vectorize.contour_plate_png(self.contours, size=200, scale_cm=17.0,
                                            label="140盐肤木-1", max_width_px=100)
        self.assertEqual(100, plate.width)
        self.assertGreater(plate.height, 100)
        # some green pixels from the leaf fill
        colours = plate.getcolors(maxcolors=100000) or []
        self.assertTrue(any(count > 1000 for count, _ in colours))


if __name__ == "__main__":
    unittest.main()
