"""Image primitives: loading, normalisation, luma, Otsu, binarisation, warping.

Owned by the Team Lead.  Faithful re-implementation of the parts of upstream
LeafByte that live in ``ImageProcessing.swift``, ``ThresholdingFilter.coreimage.metal``
and ``ImageUtils.swift``.  See ``docs/ALGORITHM.md`` §3.1-3.2 and §6.1.

Only numpy + Pillow are used (no cv2 / scipy).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

#: Luma weights, straight from the Metal kernel / vImage matrix in upstream.
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)

#: Number of histogram buckets (8 bit luma), as upstream.
HISTOGRAM_BUCKETS = 256


# --------------------------------------------------------------------------
# loading / resizing
# --------------------------------------------------------------------------
def load_image(path: str | Path) -> np.ndarray:
    """Load ``path`` as an ``(H, W, 3)`` uint8 RGB array, EXIF orientation fixed."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8).copy()


def resize_to(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to exactly ``size = (width, height)`` with bicubic resampling.

    Upstream sets ``CGContext.interpolationQuality = .high``, which is bicubic.
    """
    width, height = int(size[0]), int(size[1])
    width = max(width, 1)
    height = max(height, 1)
    if width == img.shape[1] and height == img.shape[0]:
        return img
    out = Image.fromarray(img).resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(out, dtype=np.uint8)


def resize_within(img: np.ndarray, max_dim: int = 1200) -> np.ndarray:
    """Shrink so the longest side is <= ``max_dim``, preserving aspect ratio.

    Never scales up (upstream: ``min(resizingRatio, 1)``), and floors both
    dimensions (upstream: ``roundToInt(..., rule: .down)``).
    """
    height, width = img.shape[:2]
    longest = max(width, height)
    if longest <= max_dim:
        return img
    ratio = max_dim / longest
    new_w = max(int(np.floor(width * ratio)), 1)
    new_h = max(int(np.floor(height * ratio)), 1)
    return resize_to(img, (new_w, new_h))


# --------------------------------------------------------------------------
# luma / histogram / Otsu / binarisation
# --------------------------------------------------------------------------
def luma(img: np.ndarray) -> np.ndarray:
    """Per-pixel luma in ``[0, 1]``: ``(0.299 R + 0.587 G + 0.114 B) / 255``."""
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:  # already single channel
        return arr / 255.0
    return (arr[..., 0] * _LUMA_WEIGHTS[0]
            + arr[..., 1] * _LUMA_WEIGHTS[1]
            + arr[..., 2] * _LUMA_WEIGHTS[2]) / 255.0


def luma_histogram(luma_image: np.ndarray) -> np.ndarray:
    """256-bucket histogram of 8-bit luma, matching ``vImageHistogramCalculation``.

    The 8-bit channel values are the rounded luma values in 0..255.
    """
    buckets = np.rint(np.asarray(luma_image, dtype=np.float64) * (HISTOGRAM_BUCKETS - 1))
    buckets = np.clip(buckets, 0, HISTOGRAM_BUCKETS - 1).astype(np.int64)
    return np.bincount(buckets.ravel(), minlength=HISTOGRAM_BUCKETS).astype(np.int64)


def otsu_threshold(histogram: np.ndarray) -> float:
    """Otsu's method, in the "maximise inter-class variance" form used upstream.

    Returns ``bestCut / 255`` in ``[0, 1]``.  Ties go to the *larger* bucket
    index, because upstream updates on ``>=``.
    """
    hist = np.asarray(histogram, dtype=np.float64)
    if hist.size != HISTOGRAM_BUCKETS:
        raise ValueError(f"histogram must have {HISTOGRAM_BUCKETS} buckets")
    total = hist.sum()
    if total <= 0:
        return 0.0
    indexes = np.arange(HISTOGRAM_BUCKETS, dtype=np.float64)
    sum_of_mu_numerators = float((indexes * hist).sum())

    omega0 = 0.0
    mu0_numerator = 0.0
    best_variance = 0.0
    best_cut = 0
    for index in range(HISTOGRAM_BUCKETS):
        omega0 += hist[index]
        omega1 = total - omega0
        if omega0 == 0 or omega1 == 0:
            continue
        mu0_numerator += index * hist[index]
        mu1_numerator = sum_of_mu_numerators - mu0_numerator
        mu0 = mu0_numerator / omega0
        mu1 = mu1_numerator / omega1
        variance = omega0 * omega1 * (mu0 - mu1) ** 2
        if variance >= best_variance:
            best_variance = variance
            best_cut = index
    return best_cut / (HISTOGRAM_BUCKETS - 1)


def suggested_threshold(img: np.ndarray) -> float:
    """Convenience: Otsu threshold of an RGB (or luma) image."""
    return otsu_threshold(luma_histogram(luma(img)))


def binarize(img: np.ndarray, threshold: float, background: str = "white") -> np.ndarray:
    """Boolean occupancy mask: ``True`` = foreground (leaf, scale mark, ...).

    ``background="white"`` (default): occupied where ``luma < threshold``.
    ``background="black"``: occupied where ``luma > threshold``.
    """
    lum = luma(img)
    if background == "white":
        return lum < threshold
    if background == "black":
        return lum > threshold
    raise ValueError(f"unknown background: {background!r}")


# --------------------------------------------------------------------------
# threshold diagnostics (feed the quality / confidence model)
# --------------------------------------------------------------------------
def histogram_separation(histogram: np.ndarray, cut_index: int) -> float:
    """Otsu between-class separation normalised to ``[0, 1]``.

    ``sqrt(omega0 * omega1 * (mu0 - mu1)^2 / (N^2 * total_variance))`` is the
    correlation ratio of the two classes with the pixel intensity; 1.0 means a
    perfectly bimodal (well separable) histogram, 0.0 means no information.
    """
    hist = np.asarray(histogram, dtype=np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    indexes = np.arange(HISTOGRAM_BUCKETS, dtype=np.float64)
    mean = float((indexes * hist).sum()) / total
    total_variance = float((((indexes - mean) ** 2) * hist).sum()) / total
    if total_variance <= 0:
        return 0.0
    cut_index = int(np.clip(cut_index, 0, HISTOGRAM_BUCKETS - 1))
    omega0 = float(hist[: cut_index + 1].sum())
    omega1 = float(total - omega0)
    if omega0 <= 0 or omega1 <= 0:
        return 0.0
    mu0 = float((indexes[: cut_index + 1] * hist[: cut_index + 1]).sum()) / omega0
    mu1 = float((indexes[cut_index + 1:] * hist[cut_index + 1:]).sum()) / omega1
    between = (omega0 / total) * (omega1 / total) * (mu0 - mu1) ** 2
    return float(np.sqrt(max(between, 0.0) / total_variance))


def histogram_bimodality(histogram: np.ndarray, cut_index: int, half_width: int = 8) -> float:
    """Depth of the histogram valley at the cut, in ``[0, 1]`` (1 = deep valley)."""
    hist = np.asarray(histogram, dtype=np.float64)
    if hist.sum() <= 0:
        return 0.0
    hist = hist / hist.sum()
    cut_index = int(np.clip(cut_index, 0, HISTOGRAM_BUCKETS - 1))
    lo = max(0, cut_index - half_width)
    hi = min(HISTOGRAM_BUCKETS - 1, cut_index + half_width)
    valley = float(hist[lo: hi + 1].min())
    if cut_index <= 0 or cut_index >= HISTOGRAM_BUCKETS - 1:
        return 0.0
    left_peak = float(hist[: cut_index + 1].max())
    right_peak = float(hist[cut_index:].max())
    reference = min(left_peak, right_peak)
    if reference <= 0:
        return 0.0
    return float(np.clip(1.0 - valley / reference, 0.0, 1.0))


def threshold_diagnostics(img: np.ndarray, threshold: float, band: float = 0.02) -> dict:
    """Bimodality / separation / near-threshold-mass diagnostics for a threshold."""
    lum = luma(img)
    histogram = luma_histogram(lum)
    cut_index = int(np.clip(round(threshold * (HISTOGRAM_BUCKETS - 1)), 0, HISTOGRAM_BUCKETS - 1))
    fraction_near = float(np.mean(np.abs(lum - threshold) < band))
    return {
        "separation": histogram_separation(histogram, cut_index),
        "bimodality": histogram_bimodality(histogram, cut_index),
        "near_threshold_fraction": fraction_near,
    }


# --------------------------------------------------------------------------
# perspective warping
# --------------------------------------------------------------------------
def homography_from_quad(quad, size: int) -> np.ndarray:
    """Homography mapping destination pixels -> source pixels.

    ``quad`` is ``[TL, TR, BR, BL]`` (image coordinates) and destination is the
    ``size x size`` square whose corners ``(0,0) (S,0) (S,S) (0,S)`` map onto the
    quad corners.  Returns the 3x3 matrix ``H`` with ``(x, y, 1) ~ H (u, v, 1)``.
    """
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    s = float(size)
    dst = np.array([[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]], dtype=np.float64)

    a = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    for i in range(4):
        u, v = dst[i]
        x, y = quad[i]
        a[2 * i] = [u, v, 1, 0, 0, 0, -u * x, -v * x]
        b[2 * i] = x
        a[2 * i + 1] = [0, 0, 0, u, v, 1, -u * y, -v * y]
        b[2 * i + 1] = y
    try:
        h = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:  # degenerate quad
        return np.eye(3)
    H = np.append(h, 1.0).reshape(3, 3)
    return H


def _bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample ``img`` (H, W, C) at float coordinates with bilinear interpolation."""
    height, width = img.shape[:2]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x0c = np.clip(x0, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    wx = (xs - x0)[..., None]
    wy = (ys - y0)[..., None]
    flat = img.reshape(-1, img.shape[-1]).astype(np.float64) if img.ndim == 3 else img.ravel().astype(np.float64)
    if img.ndim == 3:
        def px(yy, xx):
            return flat[yy * width + xx]
        top = px(y0c, x0c) * (1 - wx) + px(y0c, x1) * wx
        bottom = px(y1, x0c) * (1 - wx) + px(y1, x1) * wx
        out = top * (1 - wy) + bottom * wy
        return out
    top = flat[y0c * width + x0c] * (1 - wx[..., 0]) + flat[y0c * width + x1] * wx[..., 0]
    bottom = flat[y1 * width + x0c] * (1 - wx[..., 0]) + flat[y1 * width + x1] * wx[..., 0]
    return top * (1 - wy[..., 0]) + bottom * wy[..., 0]


def warp_quad(img: np.ndarray, quad, size: int) -> np.ndarray:
    """Perspective-correct the quadrilateral ``quad`` into a ``size x size`` image.

    Equivalent to upstream's ``CIPerspectiveCorrection`` + ``resizeImageIgnoringAspectRatioAndOrientation``,
    but warps the *original* image (upstream warps the already thresholded image).
    """
    size = int(size)
    H = homography_from_quad(quad, size)
    uu, vv = np.meshgrid(np.arange(size, dtype=np.float64) + 0.5,
                         np.arange(size, dtype=np.float64) + 0.5)
    denom = H[2, 0] * uu + H[2, 1] * vv + H[2, 2]
    denom[denom == 0] = 1e-12
    xs = (H[0, 0] * uu + H[0, 1] * vv + H[0, 2]) / denom
    ys = (H[1, 0] * uu + H[1, 1] * vv + H[1, 2]) / denom
    # Clamp instead of masking: outside pixels are rare after rectification and
    # sampling the border keeps the mask deterministic.
    xs = np.clip(xs, 0.0, img.shape[1] - 1.0)
    ys = np.clip(ys, 0.0, img.shape[0] - 1.0)
    out = _bilinear_sample(img, xs, ys)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def rectified_size_for_quad(quad, max_dim: int = 1200) -> int:
    """Upstream ``sizeToAdjustTo``: ``min(1200, floor(min(bbox_w, bbox_h)))``."""
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    bbox_w = quad[:, 0].max() - quad[:, 0].min()
    bbox_h = quad[:, 1].max() - quad[:, 1].min()
    return int(min(max_dim, max(1, np.floor(min(bbox_w, bbox_h)))))


# --------------------------------------------------------------------------
# visualisation
# --------------------------------------------------------------------------
def render_overlay(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    holes_mask: np.ndarray | None = None,
    quad=None,
    scale_points=None,
    max_size: int = 900,
    alpha: int = 110,
) -> Image.Image:
    """Render a review overlay: leaf in red, consumed holes in blue, scale in yellow.

    Returns a PIL ``Image`` (RGB).  Coordinates are expected in ``img``'s space.
    """
    base = Image.fromarray(img).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    def paint(m: np.ndarray | None, colour: tuple[int, int, int]) -> None:
        if m is None:
            return
        m = np.asarray(m, dtype=bool)
        if m.shape != (base.size[1], base.size[0]):
            return
        layer = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
        layer[m] = (*colour, alpha)
        overlay.alpha_composite(Image.fromarray(layer, "RGBA"))

    paint(mask, (255, 40, 40))
    paint(holes_mask, (40, 120, 255))

    draw = ImageDraw.Draw(overlay)
    if quad is not None:
        pts = [tuple(map(float, p)) for p in np.asarray(quad).reshape(-1, 2)]
        if len(pts) >= 4:
            draw.line(pts + [pts[0]], fill=(255, 220, 0, 255), width=max(2, base.size[0] // 400))
    if scale_points is not None:
        r = max(5, base.size[0] // 160)
        for px, py in np.asarray(scale_points).reshape(-1, 2):
            x, y = float(px), float(py)
            draw.line([(x - r, y - r), (x + r, y + r)], fill=(255, 220, 0, 255), width=max(2, r // 2))
            draw.line([(x - r, y + r), (x + r, y - r)], fill=(255, 220, 0, 255), width=max(2, r // 2))

    out = Image.alpha_composite(base, overlay).convert("RGB")
    if max(out.size) > max_size:
        w, h = out.size
        ratio = max_size / max(w, h)
        out = out.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.Resampling.LANCZOS)
    return out


def save_png_bytes(image: Image.Image, quality: int = 85) -> bytes:
    """Encode as JPEG bytes for compact review thumbnails."""
    import io

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
