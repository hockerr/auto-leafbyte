"""Shared data contracts for the auto-leafbyte pipeline.

This module is owned by the Team Lead.  Every other module imports its types
from here so that the pipeline stays integrable.  Keep this file small,
dependency-free (numpy only) and stable; ask the Lead before changing it.

Coordinate convention: ``(x, y)`` with ``x`` = column, ``y`` = row, origin at
the top-left pixel, exactly like ``numpy`` indexing and like the upstream
LeafByte iOS code after its CGImage/UIKit flip is undone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Label of the "outside of the image" background component, as in upstream.
BACKGROUND_LABEL = -1


@dataclass
class ComponentsInfo:
    """Result of 4-connected component labeling.

    Mirrors ``leafbyte/iOS/LeafByte/Images/ImageProcessing.swift``.

    ``label_map``
        ``int32`` array of shape ``(height, width)``.  Positive labels are
        *occupied* (foreground/leaf/scale-mark) components, negative labels are
        *empty* (background/hole) components.  Empty components that touch the
        image border are merged into :data:`BACKGROUND_LABEL` (-1).
    ``label_to_size``
        Pixel count of every component (after merging equivalent labels).
    ``label_to_member_point``
        One representative ``(x, y)`` per label; for merged classes the
        top-most member point, as upstream does.
    ``empty_to_occupied``
        For every empty label: the set of occupied labels it touches.
    """

    width: int
    height: int
    label_map: np.ndarray
    label_to_size: dict[int, int]
    label_to_member_point: dict[int, tuple[int, int]]
    empty_to_occupied: dict[int, set[int]] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------
    def occupied_labels(self) -> list[int]:
        """Occupied labels, biggest first."""
        return sorted((l for l in self.label_to_size if l > 0),
                      key=lambda l: self.label_to_size[l], reverse=True)

    def empty_labels(self) -> list[int]:
        """Empty labels except the outside-of-image background, biggest first."""
        return sorted((l for l in self.label_to_size if l < 0 and l != BACKGROUND_LABEL),
                      key=lambda l: self.label_to_size[l], reverse=True)

    def size(self, label: int) -> int:
        return self.label_to_size.get(label, 0)

    def member_point(self, label: int) -> tuple[int, int] | None:
        return self.label_to_member_point.get(label)

    def leaf_label(self) -> int | None:
        """Upstream heuristic: the largest occupied component is the leaf."""
        occ = self.occupied_labels()
        return occ[0] if occ else None


@dataclass
class RectifiedImage:
    """Perspective-corrected image produced from four scale marks.

    ``image`` is an ``(S, S, 3)`` uint8 RGB array; ``scale_px_length`` is the
    pixel length of the known scale mark *after* rectification, which for the
    upstream 4-corner-square protocol is ``S`` (the side of the square).
    """

    image: np.ndarray
    scale_px_length: float
    homography: np.ndarray | None = None
    size: int = 0


@dataclass
class ScaleDetection:
    """Output of :mod:`leafbyte.scale`."""

    found: bool
    #: Four corner points ``[(x, y)] * 4`` ordered TL, TR, BR, BL (image space).
    points: list[tuple[float, float]] = field(default_factory=list)
    #: 0..1, how much the detected quadrilateral looks like the expected square.
    quality: float = 0.0
    #: ``"quad"`` (four marks), ``"quad_relaxed"`` (looser gates),
    #: ``"quad_3dot"`` (fourth corner inferred by parallelogram completion),
    #: ``"manual"`` (reviewer supplied), ``"none"`` (not found).
    method: str = "none"
    #: All plausible mark candidates, for the review UI / debugging.
    candidates: list[dict[str, Any]] = field(default_factory=list)
    #: Human readable problems, e.g. ``"only 2 of 4 scale marks"``.
    flags: list[str] = field(default_factory=list)
    #: Free-form numbers (side lengths in px, aspect ratio, area fraction...).
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class MeasureResult:
    """Output of :mod:`leafbyte.measure`."""

    #: Total leaf area including consumed holes, in rectified pixels.
    total_px: int = 0
    #: Consumed area (holes fully enclosed by the leaf) in rectified pixels.
    consumed_px: int = 0
    leaf_label: int | None = None
    leaf_px: int = 0
    #: ``[{"label": int, "area_px": int, "centroid": (x, y)}]`` for consumed holes.
    holes: list[dict[str, Any]] = field(default_factory=list)
    #: Pixels excluded by the reviewer / user drawing.
    excluded_px: int = 0
    flags: list[str] = field(default_factory=list)
    #: QC inputs: does the leaf component touch the edge of the rectified crop?
    leaf_touches_border: bool = False
    #: QC input: second biggest occupied component / leaf size (0 = unambiguous).
    competing_ratio: float = 0.0
    #: QC input: number of consumed holes and how many of them are tiny.
    hole_count: int = 0
    tiny_holes: int = 0

    @property
    def percent_consumed(self) -> float:
        return 100.0 * self.consumed_px / self.total_px if self.total_px else 0.0
