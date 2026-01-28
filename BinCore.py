"""
3D bin-packing / box recommendation algorithm (geometry-only).

Heuristic:
- First-Fit-Decreasing by item volume.
- Bottom-left-back scan (corner positions + adaptive grid).
- Multi-SKU greedy packer: opens boxes as needed.
- Single-SKU planner: computes a good box mix using simulated capacity.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import itertools
from collections import Counter
from math import inf, ceil


# ============================================================
# 1. DATA MODELS
# ============================================================

@dataclass
class ItemType:
    """
    Represents a type of item to pack.
    Geometry-only; weight is ignored.
    """
    id: str
    length: float
    width: float
    height: float
    quantity: int = 1
    can_rotate: bool = True

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height


@dataclass
class BoxType:
    """
    Represents a type of box that can be used.

    max_boxes:
        Maximum number of boxes of this type that can be used.
        If None, there is no limit.

    cost:
        Economic cost of using one box of this type.
    """
    id: str
    inner_length: float
    inner_width: float
    inner_height: float
    cost: float
    max_boxes: Optional[int] = None

    @property
    def volume(self) -> float:
        return self.inner_length * self.inner_width * self.inner_height


@dataclass
class PlacedItem:
    """
    Concrete item instance placed in a specific box.
    """
    item_id: str
    box_id: str
    length: float
    width: float
    height: float
    position: Tuple[float, float, float]   # (x, y, z)
    rotation: Tuple[float, float, float]   # (L, W, H) after rotation


@dataclass
class BoxInstance:
    """
    Concrete box instance we are filling.
    """
    box_type: BoxType
    instance_index: int
    items: List[PlacedItem] = field(default_factory=list)

    def used_volume(self) -> float:
        return sum(it.rotation[0] * it.rotation[1] * it.rotation[2] for it in self.items)


# ============================================================
# 2. GEOMETRY HELPERS
# ============================================================

def orientations_of(item: ItemType) -> List[Tuple[float, float, float]]:
    """
    All unique orientation permutations (L, W, H) for an item.
    """
    dims = (item.length, item.width, item.height)
    return list(set(itertools.permutations(dims, 3)))


def fits_in_box(rot_dims: Tuple[float, float, float], box: BoxType) -> bool:
    """
    Check if an item with rotated dimensions rot_dims fits inside the box.
    """
    l, w, h = rot_dims
    return (l <= box.inner_length) and (w <= box.inner_width) and (h <= box.inner_height)


def aabb_overlap(a_min, a_max, b_min, b_max) -> bool:
    """
    Axis-aligned bounding box overlap test in 3D.
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
    Check if placing an item at position with dims would collide with any existing item.
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
    Float range inclusive of stop (with epsilon).
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
    Find a feasible position for dims inside box using a bottom-left-back heuristic.

    1) Try corner-like candidate positions (origin + adjacent to existing items).
    2) If none work, optionally do a coarse grid scan with adaptive resolution.
    """
    l, w, h = dims
    L = box.box_type.inner_length
    W = box.box_type.inner_width
    H = box.box_type.inner_height

    if l > L or w > W or h > H:
        return None

    # 1) Corner candidates
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
        for (x, y, z) in adj:
            if 0 <= x <= L - l + 1e-6 and 0 <= y <= W - w + 1e-6 and 0 <= z <= H - h + 1e-6:
                candidate_positions.add((round(x, 6), round(y, 6), round(z, 6)))

    for pos in sorted(candidate_positions, key=lambda p: (p[2], p[1], p[0])):
        if not collides_with_existing(box, pos, dims):
            return pos

    # 2) Grid fallback (adaptive)
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


# ============================================================
# 3. INTERNAL HELPERS
# ============================================================

def _expand_items(items: List[ItemType]) -> List[ItemType]:
    """
    Expand items with quantity>1 into a flat list of quantity=1 items.
    """
    expanded: List[ItemType] = []
    for it in items:
        for i in range(it.quantity):
            expanded.append(
                ItemType(
                    id=f"{it.id}#{i+1}",
                    length=it.length,
                    width=it.width,
                    height=it.height,
                    quantity=1,
                    can_rotate=it.can_rotate,
                )
            )
    return expanded


def _item_can_fit_in_box_type(item: ItemType, bt: BoxType) -> bool:
    oris = orientations_of(item) if item.can_rotate else [(item.length, item.width, item.height)]
    return any(fits_in_box(o, bt) for o in oris)


def _orientation_capacity_in_box(dims: Tuple[float, float, float], bt: BoxType) -> int:
    """
    Using a simple tiling estimate: how many items with 'dims' can fit in an empty box bt?
    (Axis-aligned integer tiling.)
    """
    l, w, h = dims
    if l > bt.inner_length or w > bt.inner_width or h > bt.inner_height:
        return 0
    nx = int(bt.inner_length // l)
    ny = int(bt.inner_width // w)
    nz = int(bt.inner_height // h)
    return nx * ny * nz


def _try_place_item_in_box(box: BoxInstance, item: ItemType, grid_resolution: float) -> bool:
    """
    Try to place item in box; orientations are ordered by how many copies
    of that orientation can fit in an empty box (capacity), then by fill.
    This fixes the "flat vs vertical" orientation issue.
    """
    oris = orientations_of(item) if item.can_rotate else [(item.length, item.width, item.height)]

    def orientation_score(dims):
        bt = box.box_type
        capacity = _orientation_capacity_in_box(dims, bt)
        if capacity <= 0:
            return (0, 0.0)
        l, w, h = dims
        fill = (l / bt.inner_length) * (w / bt.inner_width) * (h / bt.inner_height)
        return (capacity, fill)

    # Sort by capacity descending, then fill descending
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
            )
        )
        return True

    return False


# ============================================================
# 4. MULTI-SKU PACKER (GENERAL LIST OF ITEMS)
# ============================================================

def pack_order(
    items: List[ItemType],
    box_types: List[BoxType],
    grid_resolution: float = 1.0,
) -> Tuple[List[BoxInstance], List[ItemType]]:
    """
    Pack a list of items (possibly multiple SKUs) into available box types.

    Strategy:
    - Expand items (quantity>1 -> flat list), sort by volume desc.
    - For each item:
      1) Try existing boxes (fullest first).
      2) If not placed, open a new box chosen to roughly minimize box count
         and waste via volume-based scoring.
    - Respect max_boxes per box type.
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

    def open_new_box_for_item(item: ItemType, current_index: int) -> Optional[BoxInstance]:
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
                return (inf, inf, inf, inf)
            approx_boxes_needed = ceil(pile_vol / bt.volume)
            waste = approx_boxes_needed * bt.volume - pile_vol
            return (approx_boxes_needed, waste, bt.cost, bt.volume)

        candidates.sort(key=score)
        chosen = candidates[0]

        opened_counts[chosen.id] += 1
        idx = sum(1 for b in boxes if b.box_type.id == chosen.id) + 1
        new_box = BoxInstance(box_type=chosen, instance_index=idx)
        boxes.append(new_box)
        return new_box

    for i, item in enumerate(expanded):
        placed = False

        # Try existing boxes (fullest first)
        for box in sorted(
            boxes,
            key=lambda b: -(
                b.used_volume() / b.box_type.volume if b.box_type.volume > 0 else 0.0
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
            unassigned.append(item)

    return boxes, unassigned


# ============================================================
# 5. SINGLE-SKU PLANNER (PRECISE BOX-MIX)
# ============================================================

def _get_real_capacity_single_box(item: ItemType, box_type: BoxType, probe_resolution: float) -> int:
    """
    Estimate how many identical items fit in ONE box of box_type by simulation
    using the full 3D packer, but only a single box.
    """
    if not _item_can_fit_in_box_type(item, box_type):
        return 0

    max_theoretical = int(box_type.volume // item.volume) if item.volume > 0 else 0
    if max_theoretical <= 0:
        return 0

    dummies = [
        ItemType("DUMMY", item.length, item.width, item.height, quantity=1, can_rotate=item.can_rotate)
        for _ in range(max_theoretical)
    ]

    one_bt = BoxType(
        id="PROBE",
        inner_length=box_type.inner_length,
        inner_width=box_type.inner_width,
        inner_height=box_type.inner_height,
        cost=0.0,
        max_boxes=1,
    )

    boxes, _ = pack_order(dummies, [one_bt], grid_resolution=probe_resolution)
    if not boxes:
        return 0
    return len(boxes[0].items)


def _plan_mix_single_sku(
    quantity: int,
    box_types: List[BoxType],
    capacities: List[int],
) -> Dict[str, int]:
    """
    Plan box mix for one SKU:
    minimize (total_boxes, total_cost, total_volume).
    """
    n = len(box_types)
    best_mix: Dict[str, int] = {bt.id: 0 for bt in box_types}
    best_score = (inf, inf, inf)

    order = sorted(range(n), key=lambda k: -box_types[k].volume)

    def evaluate(counts: List[int]) -> Tuple[float, float, float]:
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

        for c in range(limit, -1, -1):
            counts[i] = c
            dfs(pos + 1, counts, covered + c * cap, used_boxes + c)
            counts[i] = 0

    dfs(0, [0] * n, covered=0, used_boxes=0)
    return best_mix


def pack_single_sku_order(
    item: ItemType,
    box_types: List[BoxType],
    grid_resolution: float = 1.0,
) -> Tuple[List[BoxInstance], List[ItemType], List[BoxType], Dict[str, int]]:
    """
    Single-SKU flow:

    1) Compute real capacity per box type (one-box simulation).
    2) Plan best mix (fewest boxes, then cost, then waste volume).
    3) Call pack_order once with those max_boxes.
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
        planned_box_types.append(
            BoxType(
                id=bt.id,
                inner_length=bt.inner_length,
                inner_width=bt.inner_width,
                inner_height=bt.inner_height,
                cost=bt.cost,
                max_boxes=planned_max,
            )
        )

    boxes, unassigned = pack_order([item], planned_box_types, grid_resolution=grid_resolution)
    return boxes, unassigned, planned_box_types, mix


# ============================================================
# 6. SUMMARY PRINTING
# ============================================================

def print_packing_summary(boxes: List[BoxInstance], unassigned_items: List[ItemType]) -> None:
    """
    Print a compact summary of boxes and item counts.
    """
    for b in boxes:
        counter = Counter(it.item_id.split("#")[0] for it in b.items)
        util = (b.used_volume() / b.box_type.volume * 100) if b.box_type.volume > 0 else 0.0
        print(f"Box {b.box_type.id}-{b.instance_index}")
        print(f" Box size: {b.box_type.inner_length}x{b.box_type.inner_width}x{b.box_type.inner_height}")
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
            print(f" - {bt.id}: used {used}, remaining: unlimited")
        else:
            remaining = max(bt.max_boxes - used, 0)
            print(f" - {bt.id}: used {used}, remaining: {remaining} (of {bt.max_boxes})")
    print()
