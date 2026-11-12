"""Sub-pixel contour extraction and vector (SVG) export.

The binary mask is the source of truth for the area numbers, but a binary mask
can only express edges as whole pixels (1 px ≈ 0.23 mm at 17 cm / 750 px), which
looks like a staircase at print resolution.  For figures we therefore trace the
**continuous** segmentation field (``ExG - cut`` or ``threshold - luma``) with
marching squares: the contour vertices sit on the field crossings, i.e. at
sub-pixel positions, so the exported path is smooth at any zoom.

Only numpy is used (no scipy / skimage).
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

Point = tuple[float, float]
Polygon = list[Point]


# ---------------------------------------------------------------------------
# marching squares
# ---------------------------------------------------------------------------
def _cell_segments(field: np.ndarray, level: float) -> list[tuple[Point, Point]]:
    """One marching-squares pass: all level-set segments of the sampled field."""
    field = np.asarray(field, dtype=np.float64)
    height, width = field.shape
    if height < 2 or width < 2:
        return []

    top_left = field[:-1, :-1]
    top_right = field[:-1, 1:]
    bottom_right = field[1:, 1:]
    bottom_left = field[1:, :-1]

    inside = (top_left > level, top_right > level, bottom_right > level, bottom_left > level)
    case = (inside[0].astype(np.uint8)
            | (inside[1].astype(np.uint8) << 1)
            | (inside[2].astype(np.uint8) << 2)
            | (inside[3].astype(np.uint8) << 3))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_top = (level - top_left) / (top_right - top_left)
        t_right = (level - top_right) / (bottom_right - top_right)
        t_bottom = (level - bottom_left) / (bottom_right - bottom_left)
        t_left = (level - top_left) / (bottom_left - top_left)
    t_top = np.nan_to_num(t_top)
    t_right = np.nan_to_num(t_right)
    t_bottom = np.nan_to_num(t_bottom)
    t_left = np.nan_to_num(t_left)

    rows, cols = np.mgrid[0:height - 1, 0:width - 1]
    # edges of each cell, as (x, y) points
    edge_points = {
        "T": np.stack([cols + t_top, rows.astype(np.float64)], axis=-1),
        "R": np.stack([cols + 1.0, rows + t_right], axis=-1),
        "B": np.stack([cols + t_bottom, rows + 1.0], axis=-1),
        "L": np.stack([cols.astype(np.float64), rows + t_left], axis=-1),
    }

    # case -> list of (edge, edge) pairs; ambiguous cases (5, 10) resolved by the
    # cell centre value (standard marching-squares disambiguation)
    table = {
        1: [("L", "T")], 2: [("T", "R")], 3: [("L", "R")], 4: [("R", "B")],
        6: [("T", "B")], 7: [("L", "B")], 8: [("B", "L")], 9: [("T", "B")],
        11: [("R", "B")], 12: [("R", "L")], 13: [("T", "R")], 14: [("L", "T")],
    }

    centre = 0.25 * (top_left + top_right + bottom_right + bottom_left)
    segments: list[tuple[Point, Point]] = []

    for value, pairs in table.items():
        mask = case == value
        if not mask.any():
            continue
        for first, second in pairs:
            points_a = edge_points[first][mask]
            points_b = edge_points[second][mask]
            segments.extend(zip(map(tuple, points_a), map(tuple, points_b)))

    # ambiguous saddle cells
    for value, pairs in ((5, [("L", "T"), ("R", "B")]), (10, [("T", "R"), ("B", "L")])):
        mask = case == value
        if not mask.any():
            continue
        flipped = centre[mask] > level
        first_points = edge_points[pairs[0][0]][mask]
        second_points = edge_points[pairs[0][1]][mask]
        third_points = edge_points[pairs[1][0]][mask]
        fourth_points = edge_points[pairs[1][1]][mask]
        for index in range(first_points.shape[0]):
            if flipped[index]:
                segments.append((tuple(first_points[index]), tuple(second_points[index])))
                segments.append((tuple(third_points[index]), tuple(fourth_points[index])))
            else:
                segments.append((tuple(first_points[index]), tuple(fourth_points[index])))
                segments.append((tuple(third_points[index]), tuple(second_points[index])))
    return segments


def _chain(segments: Iterable[tuple[Point, Point]], decimals: int = 6
           ) -> list[Polygon]:
    """Chain segments into closed polygons (dropping any open chain)."""
    def key(point: Point) -> tuple[float, float]:
        return (round(point[0], decimals), round(point[1], decimals))

    adjacency: dict[tuple[float, float], list[tuple[tuple[float, float], tuple[float, float]]]] = {}
    for start, end in segments:
        start_key, end_key = key(start), key(end)
        if start_key == end_key:
            continue
        adjacency.setdefault(start_key, []).append((start_key, end_key))
        adjacency.setdefault(end_key, []).append((end_key, start_key))

    used: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    polygons: list[Polygon] = []
    for start_key, links in adjacency.items():
        for link in links:
            if link in used:
                continue
            polygon: Polygon = [start_key]
            current = link
            closed = False
            while True:
                used.add(current)
                used.add((current[1], current[0]))
                node = current[1]
                polygon.append(node)
                if node == start_key:
                    closed = True
                    break
                candidates = [candidate for candidate in adjacency.get(node, [])
                              if (node, candidate[1]) not in used]
                if not candidates:
                    break
                # prefer a straight continuation to avoid zig-zagging between
                # coincident vertices of the same cell
                current = candidates[0]
            if closed and len(polygon) >= 4:
                polygons.append([(float(x), float(y)) for x, y in polygon[:-1]])
    return polygons


def marching_squares(field: np.ndarray, level: float = 0.0) -> list[Polygon]:
    """Closed sub-pixel contours of ``field == level`` as ``[(x, y), ...]``."""
    return _chain(_cell_segments(field, level))


def polygon_area(vertices: Polygon) -> float:
    """Signed shoelace area of a polygon (positive for counter-clockwise)."""
    if len(vertices) < 3:
        return 0.0
    points = np.asarray(vertices, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def simplify(vertices: Polygon, tolerance: float = 0.35) -> Polygon:
    """Douglas-Peucker polyline simplification (keeps the visual shape)."""
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape[0] < 4 or tolerance <= 0.0:
        return [(float(x), float(y)) for x, y in points]

    keep = np.zeros(points.shape[0], dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, points.shape[0] - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        start, end = points[first], points[last]
        segment = end - start
        length = float(np.hypot(*segment))
        if length <= 1e-12:
            distances = np.hypot(*(points[first + 1:last] - start).T)
        else:
            normal = np.array([-segment[1], segment[0]]) / length
            distances = np.abs(np.dot(points[first + 1:last] - start, normal))
        if distances.size == 0:
            continue
        index = int(np.argmax(distances))
        if distances[index] > tolerance:
            middle = first + 1 + index
            keep[middle] = True
            stack.append((first, middle))
            stack.append((middle, last))
    return [(float(x), float(y)) for x, y in points[keep]]


# ---------------------------------------------------------------------------
# SVG export
# ---------------------------------------------------------------------------
def _path_data(polygon: Polygon, decimals: int = 2) -> str:
    if not polygon:
        return ""
    head = polygon[0]
    parts = [f"M {head[0]:.{decimals}f},{head[1]:.{decimals}f}"]
    parts += [f"L {x:.{decimals}f},{y:.{decimals}f}" for x, y in polygon[1:]]
    parts.append("Z")
    return " ".join(parts)


def contour_svg(
    contours: list[Polygon],
    size: int,
    scale_cm: float,
    *,
    leaf_colour: str = "#3f7d3a",
    hole_colour: str = "#ffffff",
    outline_colour: str = "#1f4d1c",
    outline_width: float = 0.35,
    simplify_tolerance: float = 0.25,
    min_area_px: float = 12.0,
    title: str = "",
    note: str = "",
    label: str = "",
    scale_bar_cm: float = 5.0,
    band_cm: float = 2.4,
    annotate: bool = True,
) -> str:
    """Build an SVG of the leaf outline (cm units, smooth at any zoom).

    The largest contour is the leaf; smaller ones are drawn as holes.  Path
    coordinates are in rectified pixels; the ``viewBox``/``width`` mapping makes
    1 unit equal ``scale_cm / size`` cm, so the file opens at real-world size in
    Illustrator / Inkscape.
    """
    if not contours:
        return ('<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{scale_cm}cm" '
                f'height="{scale_cm}cm" viewBox="0 0 {size} {size}"></svg>\n')
    # marching squares leaves tiny spurious loops in saddle cells and on
    # near-threshold noise; drop anything below ``min_area_px`` (the area of a
    # real bite at 17 cm / 750 px is >100 px² anyway)
    ordered = sorted(contours, key=lambda polygon: abs(polygon_area(polygon)), reverse=True)
    dropped = 0
    keepers = [ordered[0]] if ordered else []
    for polygon in ordered[1:]:
        if abs(polygon_area(polygon)) >= min_area_px:
            keepers.append(polygon)
        else:
            dropped += 1
    leaf = simplify(keepers[0], simplify_tolerance) if keepers else []
    holes = [simplify(polygon, simplify_tolerance) for polygon in keepers[1:]]
    leaf_area_px = abs(polygon_area(leaf))
    hole_area_px = sum(abs(polygon_area(hole)) for hole in holes)
    units_per_pixel = scale_cm / size
    to_cm2 = units_per_pixel ** 2

    px_per_cm = size / float(scale_cm)
    band_px = float(band_cm) * px_per_cm if annotate else 0.0
    page_h_px = size + band_px
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
             f'width="{scale_cm:.4f}cm" height="{scale_cm + (band_cm if annotate else 0.0):.4f}cm" '
             f'viewBox="0 0 {size} {page_h_px:.2f}">']
    if title:
        lines.append(f"  <title>{title}</title>")
    if note:
        lines.append(f"  <desc>{note}</desc>")
    lines.append('  <g id="leaf-area">')
    lines.append(f'    <path d="{_path_data(leaf)}" fill="{leaf_colour}" fill-rule="evenodd" '
                 f'stroke="none"/>')
    lines.extend(f'    <path d="{_path_data(hole)}" fill="{hole_colour}" stroke="none"/>'
                 for hole in holes)
    lines.append("  </g>")
    lines.append('  <g id="leaf-outline" fill="none" '
                 f'stroke="{outline_colour}" stroke-width="{outline_width}" '
                 'stroke-linejoin="round" stroke-linecap="round">')
    lines.append(f'    <path d="{_path_data(leaf)}"/>')
    lines.extend(f'    <path d="{_path_data(hole)}"/>' for hole in holes)
    lines.append("  </g>")
    lines.append("  <g id=\"measurements\" font-family=\"sans-serif\" font-size=\"10\" "
                 "fill=\"#333333\">")
    lines.append(f'    <metadata data-leaf-area-cm2="{(leaf_area_px + hole_area_px) * to_cm2:.4f}" '
                 f'data-hole-area-cm2="{hole_area_px * to_cm2:.4f}" '
                 f'data-scale-cm="{scale_cm:.4f}" data-image-px="{size}" '
                 f'data-dropped-specks="{dropped}"/>')
    lines.append("  </g>")

    if annotate:
        # ---- bottom band: sample label + 0..N cm scale bar ----------------
        font_px = 0.52 * px_per_cm
        small_px = 0.38 * px_per_cm
        bar_px = float(scale_bar_cm) * px_per_cm
        bar_x = 0.5 * px_per_cm
        bar_y = size + 1.62 * px_per_cm
        bar_h = max(3.0, 0.09 * px_per_cm)
        lines.append('  <g id="annotations" font-family="Helvetica, Arial, sans-serif">')
        if label:
            lines.append(f'    <text x="{bar_x:.2f}" y="{size + 0.78 * px_per_cm:.2f}" '
                         f'font-size="{font_px:.2f}" fill="#222222">{_xml_escape(label)}</text>')
        lines.append(f'    <rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="{bar_px:.2f}" '
                     f'height="{bar_h:.2f}" fill="#222222"/>')
        # small cm ticks
        for step in range(int(round(float(scale_bar_cm))) + 1):
            tick_x = bar_x + step * px_per_cm
            tick_h = (0.26 if step in (0, int(round(float(scale_bar_cm)))) else 0.14) * px_per_cm
            lines.append(f'    <rect x="{tick_x:.2f}" y="{bar_y + bar_h:.2f}" '
                         f'width="{max(1.5, 0.03 * px_per_cm):.2f}" height="{tick_h:.2f}" fill="#222222"/>')
            if step in (0, int(round(float(scale_bar_cm)))):
                lines.append(f'    <text x="{tick_x:.2f}" y="{bar_y - 0.18 * px_per_cm:.2f}" '
                             f'font-size="{small_px:.2f}" fill="#222222">'
                             f'{"0" if step == 0 else f"{step}"}</text>')
        lines.append(f'    <text x="{bar_x + bar_px + 0.55 * px_per_cm:.2f}" '
                     f'y="{bar_y - 0.18 * px_per_cm:.2f}" font-size="{small_px:.2f}" '
                     f'fill="#222222">cm</text>')
        lines.append("  </g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# PDF export (hand written, no dependencies)
# ---------------------------------------------------------------------------
_CM_TO_PT = 72.0 / 2.54  # 28.3465 pt per cm


def _xml_escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _pdf_escape(text: str) -> str:
    return str(text).replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _label_bitmap(text: str, height_px: int = 90):
    """Render the sample label (may contain CJK) to a grayscale PIL image."""
    from PIL import Image, ImageDraw

    try:
        from leafbyte.imageops import load_cjk_font
        font = load_cjk_font(height_px)
    except Exception:  # noqa: BLE001
        font = None
    if font is None:
        return None
    probe = ImageDraw.Draw(Image.new("L", (8, 8), 255))
    try:
        box = probe.textbbox((0, 0), text, font=font)
        width = max(1, box[2] - box[0] + 8)
    except Exception:  # noqa: BLE001
        return None
    image = Image.new("L", (width, height_px + 12), 255)
    ImageDraw.Draw(image).text((4, 4), text, font=font, fill=0)
    return image


class _Pdf:
    """Minimal single-page PDF builder (vector paths + optional raster label)."""

    def __init__(self, width_pt: float, height_pt: float) -> None:
        self.width = float(width_pt)
        self.height = float(height_pt)
        self.ops: list[str] = []
        self.images: list[tuple[str, object]] = []
        self.texts: list[str] = []

    # --- drawing primitives -------------------------------------------
    def path(self, polygon, *, fill=None, stroke=None, width_pt=0.8, even_odd=True) -> None:
        if len(polygon) < 3:
            return
        first = polygon[0]
        self.ops.append(f"{first[0]:.3f} {first[1]:.3f} m")
        for x, y in polygon[1:]:
            self.ops.append(f"{x:.3f} {y:.3f} l")
        self.ops.append("h")
        if fill is not None:
            self.ops.append(f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg")
            self.ops.append("f*" if even_odd else "f")
        if stroke is not None:
            self.ops.append(f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG")
            self.ops.append(f"{width_pt:.3f} w")
            self.ops.append("S")

    def rect(self, x, y, w, h, *, fill=None, stroke=None, width_pt=0.5) -> None:
        self.ops.append(f"{x:.3f} {y:.3f} {w:.3f} {h:.3f} re")
        if fill is not None:
            self.ops.append(f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg")
            self.ops.append("f")
        if stroke is not None:
            self.ops.append(f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG")
            self.ops.append(f"{width_pt:.3f} w")
            self.ops.append("S")

    def text(self, x, y, size_pt, value, *, colour=(0.13, 0.13, 0.13)) -> None:
        self.texts.append(
            f"BT /F1 {size_pt:.2f} Tf {colour[0]:.3f} {colour[1]:.3f} {colour[2]:.3f} rg "
            f"1 0 0 1 {x:.3f} {y:.3f} Tm ({_pdf_escape(value)}) Tj ET")

    def image(self, pil_image, x, y, w, h) -> None:
        name = f"Im{len(self.images) + 1}"
        self.images.append((name, pil_image))
        self.ops.append("q")
        self.ops.append(f"{w:.3f} 0 0 {h:.3f} {x:.3f} {y:.3f} cm")
        self.ops.append(f"/{name} Do")
        self.ops.append("Q")

    # --- serialisation -------------------------------------------------
    def to_bytes(self) -> bytes:
        """Serialise to a PDF-1.4 byte string.

        Object order (cross-references must match exactly):
        1 catalog, 2 pages, 3 page, 4 content stream, 5 font, 6.. images.
        """
        content = "\n".join(self.ops + self.texts).encode("latin-1", "replace")
        catalog_id, pages_id, page_id, content_id, font_id = 1, 2, 3, 4, 5
        image_ids = [6 + index for index in range(len(self.images))]
        xobject_entries = " ".join(
            f"/{name} {object_id} 0 R"
            for (name, _image), object_id in zip(self.images, image_ids))
        resources = (f"<< /Font << /F1 {font_id} 0 R >> "
                     f"/XObject << {xobject_entries} >> >>")

        bodies: list[bytes] = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (f"<< /Type /Page /Parent 2 0 R "
             f"/MediaBox [0 0 {self.width:.3f} {self.height:.3f}] "
             f"/Resources {resources} /Contents {content_id} 0 R >>").encode("latin-1"),
            (b"<< /Length " + str(len(content)).encode("latin-1")
             + b" >>\nstream\n" + content + b"\nendstream"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        ]
        for _name, image in self.images:
            width, height = image.size
            payload = image.tobytes()
            bodies.append(
                (f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
                 f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Length {len(payload)} >>\n"
                 f"stream\n").encode("latin-1") + payload + b"\nendstream")

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets: list[int] = []
        for index, body in enumerate(bodies, start=1):
            offsets.append(len(out))
            out += f"{index} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"
        xref_offset = len(out)
        out += f"xref\n0 {len(bodies) + 1}\n".encode("latin-1")
        out += b"0000000000 65535 f \n"
        for offset in offsets:
            out += f"{offset:010d} 00000 n \n".encode("latin-1")
        out += (f"trailer\n<< /Size {len(bodies) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n").encode("latin-1")
        return bytes(out)


def verify_pdf_bytes(data: bytes) -> list[str]:
    """Structural self-check of a PDF produced by :class:`_Pdf`.

    Returns a list of problems (empty = the cross-references, stream lengths and
    object types are consistent).  Cheap insurance against the class of bug
    where an object reference points at the wrong object.
    """
    problems: list[str] = []
    if not data.startswith(b"%PDF-"):
        problems.append("missing %PDF header")
    if not data.rstrip().endswith(b"%%EOF"):
        problems.append("missing %%EOF")
    match = re.search(rb"startxref\s+(\d+)", data)
    if not match:
        return problems + ["missing startxref"]
    start = int(match.group(1))
    if data[start:start + 4] != b"xref":
        problems.append("startxref does not point at the xref table")

    # object bodies and offsets
    objects: dict[int, tuple[int, int]] = {}
    for found in re.finditer(rb"(\d+) 0 obj\n", data):
        objects[int(found.group(1))] = (found.start(), found.end())
    if not objects:
        return problems + ["no objects"]
    for index in range(1, max(objects) + 1):
        if index not in objects:
            problems.append(f"object {index} missing")
    xref_entries = re.findall(rb"(\d{10}) 00000 n", data[start:])
    if len(xref_entries) != len(objects):
        problems.append(f"xref has {len(xref_entries)} entries for {len(objects)} objects")
    for index, offset in enumerate(xref_entries, start=1):
        if not data[int(offset):].startswith(f"{index} 0 obj".encode()):
            problems.append(f"xref offset of object {index} is wrong")

    # every stream's /Length must match the bytes up to its endstream
    for found in re.finditer(rb"<< /Length (\d+) >>\nstream\n", data):
        declared = int(found.group(1))
        payload_start = found.end()
        end = data.find(b"\nendstream", payload_start)
        if end < 0:
            problems.append("stream without endstream")
            continue
        if end - payload_start != declared:
            problems.append(f"stream /Length says {declared}, actual {end - payload_start}")

    # references must resolve, and the page resources must point at the right types
    for found in re.finditer(rb"(\d+) 0 R", data):
        target = int(found.group(1))
        if target not in objects:
            problems.append(f"dangling reference {target} 0 R")
    if b"/Type /Page" not in data:
        problems.append("no page object")
    font = re.search(rb"/Font << /F1 (\d+) 0 R >>", data)
    if font and int(font.group(1)) in objects:
        start = objects[int(font.group(1))][0]
        if b"/Subtype /Type1" not in data[start:start + 200]:
            problems.append("/F1 does not point at a Type1 font")
    for name, ref in re.findall(rb"/(Im\d+) (\d+) 0 R", data):
        target = int(ref)
        snippet = data[objects[target][0]:objects[target][0] + 200] if target in objects else b""
        if b"/Subtype /Image" not in snippet:
            problems.append(f"/{name.decode()} does not point at an image XObject")
    return problems


def contour_pdf(
    contours: list[Polygon],
    size: int,
    scale_cm: float,
    *,
    label: str = "",
    scale_bar_cm: float = 5.0,
    band_cm: float = 2.4,
    annotate: bool = True,
    leaf_colour=(0.247, 0.490, 0.227),
    outline_colour=(0.122, 0.302, 0.110),
    simplify_tolerance: float = 0.25,
    min_area_px: float = 12.0,
) -> bytes:
    """Vector PDF of the leaf outline with a sample label and a 0..N cm scale bar."""
    page_w = float(scale_cm) * _CM_TO_PT
    band_pt = float(band_cm) * _CM_TO_PT if annotate else 0.0
    page_h = page_w + band_pt
    px_to_pt = page_w / float(size)

    def to_page(polygon: Polygon) -> list[tuple[float, float]]:
        return [(x * px_to_pt, page_h - y * px_to_pt) for x, y in polygon]

    pdf = _Pdf(page_w, page_h)

    ordered = sorted(contours, key=lambda item: abs(polygon_area(item)), reverse=True)
    keepers = [ordered[0]] if ordered else []
    for polygon in ordered[1:]:
        if abs(polygon_area(polygon)) >= min_area_px:
            keepers.append(polygon)
    if keepers:
        leaf = to_page(simplify(keepers[0], simplify_tolerance))
        pdf.path(leaf, fill=leaf_colour)
        for hole in keepers[1:]:
            pdf.path(to_page(simplify(hole, simplify_tolerance)), fill=(1.0, 1.0, 1.0))
        # outline: draw after the fills so the line stays visible
        pdf.path(leaf, stroke=outline_colour, width_pt=0.9)
        for hole in keepers[1:]:
            pdf.path(to_page(simplify(hole, simplify_tolerance)), stroke=outline_colour, width_pt=0.6)

    if annotate:
        margin = 0.5 * _CM_TO_PT
        font_pt = 0.42 * _CM_TO_PT
        small_pt = 0.30 * _CM_TO_PT
        bar_len = float(scale_bar_cm) * _CM_TO_PT
        bar_y = 0.55 * _CM_TO_PT
        bar_h = max(1.0, 0.08 * _CM_TO_PT)
        pdf.rect(margin, bar_y, bar_len, bar_h, fill=(0.13, 0.13, 0.13))
        for step in range(int(round(float(scale_bar_cm))) + 1):
            x = margin + step * _CM_TO_PT
            height = 0.20 * _CM_TO_PT if step in (0, int(round(float(scale_bar_cm)))) else 0.11 * _CM_TO_PT
            pdf.rect(x, bar_y + bar_h, max(0.6, 0.025 * _CM_TO_PT), height,
                     fill=(0.13, 0.13, 0.13))
            if step in (0, int(round(float(scale_bar_cm)))):
                pdf.text(x, bar_y + bar_h + 0.24 * _CM_TO_PT, small_pt,
                         "0" if step == 0 else str(step))
        pdf.text(margin + bar_len + 0.5 * _CM_TO_PT, bar_y + bar_h + 0.24 * _CM_TO_PT,
                 small_pt, "cm")
        if label:
            bitmap = _label_bitmap(label, height_px=110)
            if bitmap is not None:
                width_pt = font_pt * bitmap.size[0] / max(bitmap.size[1], 1)
                pdf.image(bitmap, margin, 1.45 * _CM_TO_PT, width_pt, 1.05 * _CM_TO_PT)
            else:  # pragma: no cover - only without a CJK font
                pdf.text(margin, 1.45 * _CM_TO_PT, font_pt, label)
    return pdf.to_bytes()


# ---------------------------------------------------------------------------
# raster plate (PNG) with the same annotations, for slides / quick checks
# ---------------------------------------------------------------------------
def contour_plate_png(
    contours: list[Polygon],
    size: int,
    scale_cm: float,
    *,
    label: str = "",
    scale_bar_cm: float = 5.0,
    band_cm: float = 2.4,
    annotate: bool = True,
    max_width_px: int = 1000,
    leaf_fill=(150, 200, 140),
    outline_colour=(30, 90, 30),
    simplify_tolerance: float = 0.25,
    min_area_px: float = 12.0,
):
    """Rasterised version of :func:`contour_svg` (PIL, CJK labels supported)."""
    from PIL import Image, ImageDraw

    px_per_cm = float(size) / float(scale_cm)
    band_px = int(round(float(band_cm) * px_per_cm)) if annotate else 0
    width = int(size)
    height = int(size) + band_px
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    ordered = sorted(contours, key=lambda item: abs(polygon_area(item)), reverse=True)
    keepers = [ordered[0]] if ordered else []
    for polygon in ordered[1:]:
        if abs(polygon_area(polygon)) >= min_area_px:
            keepers.append(polygon)
    if keepers:
        leaf = simplify(keepers[0], simplify_tolerance)
        draw.polygon([(float(x), float(y)) for x, y in leaf], fill=leaf_fill,
                     outline=outline_colour)
        for hole in keepers[1:]:
            polygon = simplify(hole, simplify_tolerance)
            draw.polygon([(float(x), float(y)) for x, y in polygon], fill=(255, 255, 255),
                         outline=outline_colour)

    if annotate:
        try:
            from leafbyte.imageops import load_cjk_font
            label_font = load_cjk_font(int(0.52 * px_per_cm))
            small_font = load_cjk_font(int(0.36 * px_per_cm))
        except Exception:  # noqa: BLE001
            label_font = small_font = None
        margin = 0.5 * px_per_cm
        bar_len = float(scale_bar_cm) * px_per_cm
        bar_y = height - 1.15 * px_per_cm
        bar_h = max(2, 0.08 * px_per_cm)
        draw.rectangle([margin, bar_y, margin + bar_len, bar_y + bar_h], fill=(30, 30, 30))
        for step in range(int(round(float(scale_bar_cm))) + 1):
            x = margin + step * px_per_cm
            tick_h = (0.24 if step in (0, int(round(float(scale_bar_cm)))) else 0.13) * px_per_cm
            draw.rectangle([x, bar_y + bar_h, x + max(1, 0.03 * px_per_cm), bar_y + bar_h + tick_h],
                           fill=(30, 30, 30))
            if step in (0, int(round(float(scale_bar_cm)))):
                draw.text((x, bar_y + bar_h + tick_h + 0.05 * px_per_cm),
                          "0" if step == 0 else str(step), fill=(30, 30, 30), font=small_font)
        draw.text((margin + bar_len + 0.55 * px_per_cm, bar_y + bar_h + 0.15 * px_per_cm),
                  "cm", fill=(30, 30, 30), font=small_font)
        if label:
            draw.text((margin, size + 0.55 * px_per_cm), label, fill=(20, 20, 20), font=label_font)

    if max_width_px and image.width > max_width_px:
        ratio = max_width_px / image.width
        image = image.resize((max_width_px, int(round(image.height * ratio))),
                             Image.Resampling.LANCZOS)
    return image
