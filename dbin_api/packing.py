# dbin_api/packing.py
"""
Core packing logic extracted and adapted from the original BinCore.py.

This module provides the geometry-only 3D packing algorithms and helper
functions used by the API:
- orientations_of, fits_in_box, collision tests
- find_feasible_position search
- multi-SKU packer: pack_order
- single-SKU planner: _get_real_capacity_single_box, _plan_mix_single_sku, pack_single_sku_order
- summary printing helpers

The implementations intentionally remain pure-Python and operate on the
dataclasses defined in `dbin_api.models`.
"""

from __future__ import annotations

import itertools
from collections import Counter
from math import ceil, inf
from typing import Dict, List, Optional, Tuple

from .models import BoxInstance, BoxType, ItemType, PlacedItem

# ----------------------------
# Geometry helpers
# ----------------------------


def orientations_of(item: ItemType) -> List[Tuple[float, float, float]]:
    """
    Return unique axis-aligned orientation permutations (L, W, H) for an item.
    """
    dims = (item.length, item.width, item.height)
    # set to remove duplicates when dimensions equal
    return list(set(itertools.permutations(dims, 3)))


def fits_in_box(rot_dims: Tuple[float, float, float], box: BoxType) -> bool:
    """
    Check if rotated dimensions fit in the box interior.
    """
    l, w, h = rot_dims
    return (
        (l <= box.inner_length) and (w <= box.inner_width) and (h <= box.inner_height)
    )


def aabb_overlap(a_min, a_max, b_min, b_max) -> bool:
    """
    Axis-aligned bounding box overlap test in 3D.
    Returns True if boxes overlap (i.e., collision).
    """
    for i in range(3):
        if a_max[i] <= b_min[i] or b_max[i] <= a_min[i]:
            return False
    return True


def collides_with_existing(
    box: BoxInstance,
    position: Tuple[float, float, float],
    dims: Tuple[float, float, float],
) -> bool:
    """
    Return True if an item of size `dims` placed at `position` would collide
    with any item already in `box`.
    """
    x, y, z = position
    l, w, h = dims
    new_min = (x, y, z)
    new_max = (x + l, y + w, z + h)

    for placed in box.items:
        px, py, pz = placed.position
        pl, pw, ph = placed.rotation
        old_min = (px, py, pz)
        old_max = (px + pl, py + pw, pz + ph)
        if aabb_overlap(new_min, new_max, old_min, old_max):
            return True

    return False


def frange(start: float, stop: float, step: float):
    """
    Floating range generator inclusive of stop (with small epsilon).
    """
    v = start
    while v <= stop + 1e-9:
        yield v
        v += step


def find_feasible_position(
    box: BoxInstance,
    dims: Tuple[float, float, float],
    resolution: float = 1.0,
    max_grid_points: int = 8000,
) -> Optional[Tuple[float, float, float]]:
    """
    Bottom-left-back heuristic for finding a feasible placement position in `box`.

    1) Try "corner" candidates derived from existing items (origin + adjacent positions).
    2) If none found, perform a coarse grid scan with adaptively increased resolution
       if the total grid would be too large.
    """
    l, w, h = dims
    L = box.box_type.inner_length
    W = box.box_type.inner_width
    H = box.box_type.inner_height

    if l > L or w > W or h > H:
        return None

    # Candidate corner-like positions
    candidate_positions = set()
    candidate_positions.add((0.0, 0.0, 0.0))

    for placed in box.items:
        px, py, pz = placed.position
        pl, pw, ph = placed.rotation
        adj = [
            (px + pl, py, pz),
            (px, py + pw, pz),
            (px, py, pz + ph),
            (px - l, py, pz),
            (px, py - w, pz),
            (px, py, pz - h),
        ]
        for x, y, z in adj:
            if (
                0 <= x <= L - l + 1e-6
                and 0 <= y <= W - w + 1e-6
                and 0 <= z <= H - h + 1e-6
            ):
                candidate_positions.add((round(x, 6), round(y, 6), round(z, 6)))

    # Sort by z, then y, then x (bottom-left-back priority)
    for pos in sorted(candidate_positions, key=lambda p: (p[2], p[1], p[0])):
        if not collides_with_existing(box, pos, dims):
            return pos

    # Grid fallback
    if resolution <= 0:
        return None

    nx = int((L - l) // resolution) + 1
    ny = int((W - w) // resolution) + 1
    nz = int((H - h) // resolution) + 1
    total_points = max(nx, 0) * max(ny, 0) * max(nz, 0)

    if total_points > max_grid_points and total_points > 0:
        factor = (total_points / max_grid_points) ** (1 / 3)
        resolution *= factor

    xs = [round(v, 6) for v in frange(0.0, L - l + 1e-6, resolution)]
    ys = [round(v, 6) for v in frange(0.0, W - w + 1e-6, resolution)]
    zs = [round(v, 6) for v in frange(0.0, H - h + 1e-6, resolution)]

    for z in zs:
        for y in ys:
            for x in xs:
                pos = (x, y, z)
                if pos in candidate_positions:
                    continue
                if not collides_with_existing(box, pos, dims):
                    return pos

    return None


# ----------------------------
# Internal helpers for packing
# ----------------------------


def _expand_items(items: List[ItemType]) -> List[ItemType]:
    """
    Expand items with quantity > 1 into a flat list of individual ItemType objects.
    Each expanded item receives an id suffix: '<original_id>#<index>'.
    """
    expanded: List[ItemType] = []
    for it in items:
        for i in range(it.quantity):
            expanded.append(
                ItemType(
                    id=f"{it.id}#{i + 1}",
                    length=it.length,
                    width=it.width,
                    height=it.height,
                    quantity=1,
                    can_rotate=it.can_rotate,
                    name=it.name,
                )
            )
    return expanded


def _item_can_fit_in_box_type(item: ItemType, bt: BoxType) -> bool:
    oris = (
        orientations_of(item)
        if item.can_rotate
        else [(item.length, item.width, item.height)]
    )
    return any(fits_in_box(o, bt) for o in oris)


def _orientation_capacity_in_box(dims: Tuple[float, float, float], bt: BoxType) -> int:
    """
    Estimate how many items of size `dims` can be tiled into an empty box `bt`
    using integer floor tiling per axis.
    """
    l, w, h = dims
    if l > bt.inner_length or w > bt.inner_width or h > bt.inner_height:
        return 0
    nx = int(bt.inner_length // l)
    ny = int(bt.inner_width // w)
    nz = int(bt.inner_height // h)
    return nx * ny * nz


def _try_place_item_in_box(
    box: BoxInstance, item: ItemType, grid_resolution: float
) -> bool:
    """
    Attempt to place `item` into `box`. Returns True on success (the box is mutated).
    Orientation order aims to prefer orientations that tile more copies into an empty box,
    then higher fill factor.
    """
    # Respect effective_volume limits if set
    if box.box_type.effective_volume is not None:
        current_volume = box.used_volume()
        item_volume = item.volume
        if current_volume + item_volume > box.box_type.effective_volume:
            return False

    oris = (
        orientations_of(item)
        if item.can_rotate
        else [(item.length, item.width, item.height)]
    )

    def orientation_score(dims: Tuple[float, float, float]):
        bt = box.box_type
        capacity = _orientation_capacity_in_box(dims, bt)
        if capacity <= 0:
            return (0, 0.0)
        l, w, h = dims
        fill = (l / bt.inner_length) * (w / bt.inner_width) * (h / bt.inner_height)
        return (capacity, fill)

    # Sort orientations by capacity then fill (descending)
    oris.sort(key=orientation_score, reverse=True)

    for dims in oris:
        if not fits_in_box(dims, box.box_type):
            continue
        pos = find_feasible_position(box, dims, resolution=grid_resolution)
        if pos is None:
            continue

        box.items.append(
            PlacedItem(
                item_id=item.id,
                box_id=f"{box.box_type.id}-{box.instance_index}",
                length=item.length,
                width=item.width,
                height=item.height,
                position=pos,
                rotation=dims,
                item_name=item.name,  # pyright: ignore[reportArgumentType]
            )
        )
        return True

    return False


# ----------------------------
# Multi-SKU packer
# ----------------------------


def pack_order(
    items: List[ItemType], box_types: List[BoxType], grid_resolution: float = 1.0
) -> Tuple[List[BoxInstance], List[ItemType]]:
    """
    Pack a list of items (multi-SKU or expanded single-SKU) into available box types.

    Returns a tuple (boxes, unassigned_items).
    """
    expanded = _expand_items(items)
    expanded.sort(key=lambda it: it.volume, reverse=True)

    box_types_list = list(box_types)
    opened_counts: Dict[str, int] = {bt.id: 0 for bt in box_types_list}

    boxes: List[BoxInstance] = []
    unassigned: List[ItemType] = []

    def pile_volume_from(idx: int) -> float:
        return sum(it.volume for it in expanded[idx:])

    def can_open(bt: BoxType) -> bool:
        return (bt.max_boxes is None) or (opened_counts[bt.id] < bt.max_boxes)

    def open_new_box_for_item(
        item: ItemType, current_index: int
    ) -> Optional[BoxInstance]:
        pile_vol = pile_volume_from(current_index)
        candidates: List[BoxType] = []

        for bt in box_types_list:
            if not can_open(bt):
                continue
            if _item_can_fit_in_box_type(item, bt):
                candidates.append(bt)

        if not candidates:
            return None

        def score(bt: BoxType):
            if bt.volume <= 0:
                return (inf, inf, inf)
            approx_boxes_needed = ceil(pile_vol / bt.volume)
            total_volume = approx_boxes_needed * bt.volume
            waste_ratio = 1 - (pile_vol / total_volume) if total_volume > 0 else 1.0
            total_cost = approx_boxes_needed * bt.cost
            waste_volume = total_volume - pile_vol
            # Cubic penalty on waste to prefer tighter fits
            waste_penalty = 1 + 3 * waste_ratio**3
            effective_boxes = approx_boxes_needed * waste_penalty
            return (effective_boxes, total_cost, waste_volume)

        candidates.sort(key=score)
        chosen = candidates[0]

        opened_counts[chosen.id] += 1
        idx = sum(1 for b in boxes if b.box_type.id == chosen.id) + 1
        new_box = BoxInstance(box_type=chosen, instance_index=idx)
        boxes.append(new_box)
        return new_box

    for i, item in enumerate(expanded):
        placed = False

        # Try existing boxes, prefer those with higher utilization
        for box in sorted(
            boxes,
            key=lambda b: (
                -(b.used_volume() / b.box_type.volume if b.box_type.volume > 0 else 0.0)
            ),
        ):
            if _try_place_item_in_box(box, item, grid_resolution):
                placed = True
                break

        if placed:
            continue

        new_box = open_new_box_for_item(item, i)
        if new_box is None:
            unassigned.append(item)
            continue

        if not _try_place_item_in_box(new_box, item, grid_resolution):
            # Could not place even after opening box
            unassigned.append(item)

    # Remove empty boxes (if any)
    boxes = [b for b in boxes if b.items]

    return boxes, unassigned


# ----------------------------
# Single-SKU planner
# ----------------------------


def _get_real_capacity_single_box(
    item: ItemType, box_type: BoxType, probe_resolution: float
) -> int:
    """
    Estimate how many identical `item` can fit in a single `box_type` by simulating
    packing of up to theoretical maximum items into one box.
    """
    if not _item_can_fit_in_box_type(item, box_type):
        return 0

    max_theoretical = int(box_type.volume // item.volume) if item.volume > 0 else 0
    if max_theoretical <= 0:
        return 0

    # Create dummy items to probe real capacity (each with quantity 1)
    dummies = [
        ItemType(
            "DUMMY",
            item.length,
            item.width,
            item.height,
            quantity=1,
            can_rotate=item.can_rotate,
            name="DUMMY",
        )
        for _ in range(max_theoretical)
    ]

    # Create a probe box type with max_boxes=1 and same dimensions
    probe_bt = BoxType(
        id="PROBE",
        inner_length=box_type.inner_length,
        inner_width=box_type.inner_width,
        inner_height=box_type.inner_height,
        cost=0.0,
        max_boxes=1,
        effective_volume=box_type.effective_volume,
        container_type=box_type.container_type,
        name="PROBE",
    )

    boxes, _ = pack_order(dummies, [probe_bt], grid_resolution=probe_resolution)
    if not boxes:
        return 0
    return len(boxes[0].items)


def _plan_mix_single_sku(
    quantity: int, box_types: List[BoxType], capacities: List[int]
) -> Dict[str, int]:
    """
    Given the per-box capacities for a single SKU and desired `quantity`, find a mix
    (how many boxes of each type) minimizing lexicographically:
    (total_boxes, total_cost, total_volume).
    """
    n = len(box_types)
    best_mix: Dict[str, int] = {bt.id: 0 for bt in box_types}
    best_score = (inf, inf, inf)

    # Order indices by descending box volume (heuristic)
    order = sorted(range(n), key=lambda k: -box_types[k].volume)

    def evaluate(counts: List[int]):
        total_boxes = sum(counts)
        total_cost = sum(counts[i] * box_types[i].cost for i in range(n))
        total_vol = sum(counts[i] * box_types[i].volume for i in range(n))
        return (total_boxes, total_cost, total_vol)

    def dfs(pos: int, counts: List[int], covered: int, used_boxes: int):
        nonlocal best_mix, best_score

        if used_boxes > best_score[0]:
            return

        if covered >= quantity:
            score = evaluate(counts)
            if score < best_score:
                best_score = score
                best_mix = {box_types[i].id: counts[i] for i in range(n)}
            return

        if pos >= len(order):
            return

        i = order[pos]
        cap = capacities[i]
        bt = box_types[i]

        if cap <= 0:
            dfs(pos + 1, counts, covered, used_boxes)
            return

        max_use = bt.max_boxes if bt.max_boxes is not None else quantity
        needed = ceil((quantity - covered) / cap)
        limit = min(max_use, needed)

        # try using from limit down to 0 boxes of type i
        for c in range(limit, -1, -1):
            counts[i] = c
            dfs(pos + 1, counts, covered + c * cap, used_boxes + c)
            counts[i] = 0

    dfs(0, [0] * n, covered=0, used_boxes=0)
    return best_mix


def pack_single_sku_order(
    item: ItemType, box_types: List[BoxType], grid_resolution: float = 1.0
) -> Tuple[List[BoxInstance], List[ItemType], List[BoxType], Dict[str, int]]:
    """
    Single-SKU packing flow:
    1) Probe each box type to estimate how many identical items can fit in one box.
    2) Plan a box mix to cover the total item quantity using the capacities.
    3) Call the general packer with max_boxes set according to the planned mix.

    Returns: (boxes, unassigned_items, planned_box_types, mix)
    """
    probe_resolution = min(item.length, item.width, item.height)

    capacities = [
        _get_real_capacity_single_box(item, bt, probe_resolution=probe_resolution)
        for bt in box_types
    ]

    mix = _plan_mix_single_sku(item.quantity, box_types, capacities)

    planned_box_types: List[BoxType] = []
    for bt in box_types:
        planned_max = mix.get(bt.id, 0)
        if bt.max_boxes is not None:
            planned_max = min(planned_max, bt.max_boxes)
        # create a shallow copy with max_boxes overridden for the planning run
        planned_box_types.append(
            BoxType(
                id=bt.id,
                inner_length=bt.inner_length,
                inner_width=bt.inner_width,
                inner_height=bt.inner_height,
                cost=bt.cost,
                max_boxes=planned_max,
                effective_volume=bt.effective_volume,
                container_type=bt.container_type,
                name=bt.name,
            )
        )

    boxes, unassigned = pack_order(
        [item], planned_box_types, grid_resolution=grid_resolution
    )
    return boxes, unassigned, planned_box_types, mix


# ----------------------------
# Summary printing helpers
# ----------------------------


def print_packing_summary(
    boxes: List[BoxInstance], unassigned_items: List[ItemType]
) -> None:
    """
    Print a human-friendly packing summary to stdout.
    """
    for b in boxes:
        counter = Counter(it.item_id.split("#")[0] for it in b.items)
        util = (
            (b.used_volume() / b.box_type.volume * 100)
            if b.box_type.volume > 0
            else 0.0
        )
        print(f"Box {b.box_type.name}-{b.instance_index}")
        print(
            f" Box size: {b.box_type.inner_length}x{b.box_type.inner_width}x{b.box_type.inner_height}"
        )
        print(f" Volume utilization: {util:.1f}%")
        print(" Items:")
        for item_id, qty in counter.items():
            print(f" - {item_id}: {qty}")
        print()

    if unassigned_items:
        counter_un = Counter(it.id.split("#")[0] for it in unassigned_items)
        print("Items that could NOT be packed:")
        for item_id, qty in counter_un.items():
            print(f" - {item_id}: {qty}")
    else:
        print("All items were successfully packed.")


def print_box_stock_usage(box_types: List[BoxType], boxes: List[BoxInstance]) -> None:
    """
    Print how many boxes of each type were used and remaining stock.
    """
    used_counts = Counter(b.box_type.id for b in boxes)
    print("Box stock usage:")
    for bt in box_types:
        used = used_counts.get(bt.id, 0)
        if bt.max_boxes is None:
            print(f" - {bt.name}: used {used}, remaining: unlimited")
        else:
            remaining = max(bt.max_boxes - used, 0)
            print(
                f" - {bt.name}: used {used}, remaining: {remaining} (of {bt.max_boxes})"
            )
    print()


__all__ = [
    "orientations_of",
    "fits_in_box",
    "aabb_overlap",
    "collides_with_existing",
    "find_feasible_position",
    "pack_order",
    "pack_single_sku_order",
    "print_packing_summary",
    "print_box_stock_usage",
]
