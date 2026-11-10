"""4-connected component labeling, faithful to upstream ``labelConnectedComponents``.

Owned by the Team Lead.  See ``docs/ALGORITHM.md`` §3.3.

The Swift implementation walks every pixel and merges labels with a union-find.
A per-pixel Python port would take minutes on a 1200x1200 photo, so this version
labels *runs* of equal occupancy per scan line and merges vertically overlapping
runs.  The resulting partition, component sizes, member points, background
semantics and empty->occupied adjacency are identical; only the internal label
numbering differs, and labels are canonicalised to one representative per
component (``max`` of the class, like upstream).
"""

from __future__ import annotations

import numpy as np

from .types import BACKGROUND_LABEL, ComponentsInfo


class _UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def make(self, x: int) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: int) -> int:
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


def _row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Maximal ``True`` runs of one scan line as inclusive ``(x0, x1)`` pairs."""
    padded = np.concatenate(([False], np.asarray(row, dtype=bool), [False]))
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    if transitions.size == 0:
        return []
    return [(int(transitions[i]), int(transitions[i + 1] - 1)) for i in range(0, transitions.size, 2)]


def _overlapping_labels(
    previous_runs: list[tuple[int, int, int]], current_runs: list[tuple[int, int]]
) -> list[list[int]]:
    """For each current run, labels of vertically overlapping previous-row runs."""
    result: list[list[int]] = []
    count = len(previous_runs)
    cursor = 0
    for current_start, current_end in current_runs:
        while cursor < count and previous_runs[cursor][1] < current_start:
            cursor += 1
        probe = cursor
        labels: list[int] = []
        while probe < count and previous_runs[probe][0] <= current_end:
            labels.append(previous_runs[probe][2])
            probe += 1
        result.append(labels)
    return result


def _shifted(labels: np.ndarray, dy: int, dx: int, fill: int = 0) -> np.ndarray:
    """``out[y, x] = labels[y + dy, x + dx]`` with ``fill`` outside the image."""
    height, width = labels.shape
    out = np.full((height, width), fill, dtype=labels.dtype)
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    out[src_y0 + dy:src_y1 + dy, src_x0 + dx:src_x1 + dx] = labels[src_y0:src_y1, src_x0:src_x1]
    return out


def label_components(mask: np.ndarray) -> ComponentsInfo:
    """Label the 4-connected components of ``mask`` (``True`` = occupied).

    Positive labels are occupied components, negative labels are empty
    components; empty pixels touching the image border share
    :data:`~leafbyte.types.BACKGROUND_LABEL`.  Every returned label is the
    canonical representative of its component.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D")
    height, width = mask.shape
    if height == 0 or width == 0:
        return ComponentsInfo(
            width=width,
            height=height,
            label_map=np.zeros((height, width), np.int32),
            label_to_size={},
            label_to_member_point={},
            empty_to_occupied={},
        )

    uf = _UnionFind()
    uf.make(BACKGROUND_LABEL)
    background_has_members = False

    next_occupied_label = 1
    next_empty_label = -2
    provisional_size: dict[int, int] = {}
    provisional_member: dict[int, tuple[int, int]] = {}

    # Runs with their provisional label, per row, kept for painting the label map.
    occupied_runs_per_row: list[list[tuple[int, int, int]]] = []
    empty_runs_per_row: list[list[tuple[int, int, int]]] = []

    previous_occupied: list[tuple[int, int, int]] = []
    previous_empty: list[tuple[int, int, int]] = []

    for y in range(height):
        row = mask[y]
        occupied_runs = _row_runs(row)
        empty_runs = _row_runs(~row)

        current_occupied: list[tuple[int, int, int]] = []
        for (x0, x1), overlapping in zip(occupied_runs, _overlapping_labels(previous_occupied, occupied_runs)):
            if overlapping:
                label = overlapping[0]
                for other in overlapping[1:]:
                    uf.union(label, other)
            else:
                label = next_occupied_label
                next_occupied_label += 1
                uf.make(label)
                provisional_size[label] = 0
                provisional_member[label] = (x0, y)
            provisional_size[label] = provisional_size.get(label, 0) + (x1 - x0 + 1)
            current_occupied.append((x0, x1, label))

        current_empty: list[tuple[int, int, int]] = []
        for (x0, x1), overlapping in zip(empty_runs, _overlapping_labels(previous_empty, empty_runs)):
            if overlapping:
                label = overlapping[0]
                for other in overlapping[1:]:
                    uf.union(label, other)
            else:
                label = next_empty_label
                next_empty_label -= 1
                uf.make(label)
                provisional_size[label] = 0
                provisional_member[label] = (x0, y)
            provisional_size[label] = provisional_size.get(label, 0) + (x1 - x0 + 1)
            # Any empty pixel on the border belongs to the outside-of-image class.
            if y == 0 or y == height - 1 or x0 == 0 or x1 == width - 1:
                uf.union(label, BACKGROUND_LABEL)
                background_has_members = True
            current_empty.append((x0, x1, label))

        occupied_runs_per_row.append(current_occupied)
        empty_runs_per_row.append(current_empty)
        previous_occupied = current_occupied
        previous_empty = current_empty

    # ---- canonicalise the components -------------------------------------
    classes: dict[int, list[int]] = {}
    for label in provisional_size:
        classes.setdefault(uf.find(label), []).append(label)
    if background_has_members:
        background_class = uf.find(BACKGROUND_LABEL)
        classes.setdefault(background_class, [])
        if BACKGROUND_LABEL not in classes[background_class]:
            classes[background_class].append(BACKGROUND_LABEL)

    label_to_size: dict[int, int] = {}
    label_to_member_point: dict[int, tuple[int, int]] = {}
    representative: dict[int, int] = {}
    for members in classes.values():
        # Upstream takes the max label of each class as representative, which
        # makes the outside-of-image class come out as BACKGROUND_LABEL (-1).
        rep = max(members)
        total = sum(provisional_size.get(m, 0) for m in members)
        points = [provisional_member[m] for m in members if m in provisional_member]
        member = min(points, key=lambda p: (p[1], p[0])) if points else (0, 0)
        label_to_size[rep] = total
        label_to_member_point[rep] = member
        for m in members:
            representative[m] = rep

    # ---- label map -------------------------------------------------------
    label_map = np.zeros((height, width), dtype=np.int32)
    for y in range(height):
        for x0, x1, label in occupied_runs_per_row[y]:
            label_map[y, x0:x1 + 1] = representative[label]
        for x0, x1, label in empty_runs_per_row[y]:
            label_map[y, x0:x1 + 1] = representative[label]

    # ---- empty -> occupied adjacency -------------------------------------
    empty_to_occupied: dict[int, set[int]] = {}
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbour = _shifted(label_map, dy, dx, fill=0)
        interesting = (label_map < 0) & (neighbour > 0)
        if not interesting.any():
            continue
        pairs = np.unique(np.stack([label_map[interesting], neighbour[interesting]], axis=1), axis=0)
        for empty_label, occupied_label in pairs:
            empty_to_occupied.setdefault(int(empty_label), set()).add(int(occupied_label))
    for label in label_to_size:
        if label < 0:
            empty_to_occupied.setdefault(label, set())

    return ComponentsInfo(
        width=width,
        height=height,
        label_map=label_map,
        label_to_size=label_to_size,
        label_to_member_point=label_to_member_point,
        empty_to_occupied=empty_to_occupied,
    )
