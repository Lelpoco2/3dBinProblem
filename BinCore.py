"""
3D bin-packing / box recommendation algorithm (geometry-only).

Main features
- 3D placement with a bottom-left-back heuristic (try "corner" candidate points first).
- Adaptive grid fallback to avoid exploding runtimes with very small grid_resolution.
- Multi-item (multi-SKU) greedy packing: open boxes as needed, respecting max_boxes.
- Single-SKU planner: compute a good box mix using "real capacity" (fast simulation per box type),
  then pack with those limits.

Notes
- This is a heuristic 3D packer: exact optimal packing is NP-hard.
- Units are whatever you use consistently (cm/mm/etc.).
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
    Item definition. 'quantity' can be > 1 (it will be expanded in the packers).
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
    Box definition.
    max_boxes: stock limit for this box type (None = unlimited).
    cost: arbitrary cost (material, shipping, etc.).
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
    rotation holds the used (L,W,H) after rotation.
    """
    item_id: str
    box_id: str
    length: float
    width: float
    height: float
    position: Tuple[float, float, float]   # (x, y, z) min corner
    rotation: Tuple[float, float, float]   # (L, W, H) after rotation


@dataclass
class BoxInstance:
    """
    Concrete box instance that we fill with PlacedItem.
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
    l, w, h = rot_dims
    return (l <= box.inner_length) and (w <= box.inner_width) and (h <= box.inner_height)


def aabb_overlap(a_min, a_max, b_min, b_max) -> bool:
    """
    Axis-aligned bounding box overlap test in 3D.
    True means overlap; False means separated.
    """
    for i in range(3):
        if a_max[i] <= b_min[i] or b_max[i] <= a_min[i]:
            return False
    return True


def collides_with_existing(
    box: BoxInstance,
    position: Tuple[float, float, float],
    dims: Tuple[float, float, float]
) -> bool:
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
    Find a feasible (x,y,z) position for dims inside 'box', avoiding collisions.

    Strategy:
    1) Try a small set of "corner" candidate positions:
       - origin
       - adjacent to each placed item (right/behind/top/left/front/bottom)
    2) If that fails, optionally try a grid scan, but with an adaptive resolution
       so the number of points doesn't explode.
    """
    l, w, h = dims
    L = box.box_type.inner_length
    W = box.box_type.inner_width
    H = box.box_type.inner_height

    if l > L or w > W or h > H:
        return None

    # --- 1) Corner candidates (fast) ---
    candidate_positions = set()
    candidate_positions.add((0.0, 0.0, 0.0))

    for placed in box.items:
        px, py, pz = placed.position
        pl, pw, ph = placed.rotation

        adj = [
            (px + pl, py, pz),      # right
            (px, py + pw, pz),      # behind
            (px, py, pz + ph),      # top
            (px - l, py, pz),       # left
            (px, py - w, pz),       # front
            (px, py, pz - h),       # below
        ]

        for (x, y, z) in adj:
            if 0 <= x <= L - l + 1e-6 and 0 <= y <= W - w + 1e-6 and 0 <= z <= H - h + 1e-6:
                candidate_positions.add((round(x, 6), round(y, 6), round(z, 6)))

    # bottom-left-back priority
    for pos in sorted(candidate_positions, key=lambda p: (p[2], p[1], p[0])):
        if not collides_with_existing(box, pos, dims):
            return pos

    # --- 2) Grid fallback (adaptive) ---
    if resolution <= 0:
        return None

    # Estimate point count; coarsen if needed
    nx = int((L - l) // resolution) + 1
    ny = int((W - w) // resolution) + 1
    nz = int((H - h) // resolution) + 1
    total_points = max(nx, 0) * max(ny, 0) * max(nz, 0)

    if total_points > max_grid_points and total_points > 0:
        # Increase resolution so total_points becomes ~max_grid_points
        factor = (total_points / max_grid_points) ** (1 / 3)
        resolution *= factor

    xs = [round(v, 6) for v in frange(0.0, L - l + 1e-6, resolution)]
    ys = [round(v, 6) for v in frange(0.0, W - w + 1e-6, resolution)]
    zs = [round(v, 6) for v in frange(0.0, H - h + 1e-6, resolution)]

    # Iterate in bottom-left-back order (z, y, x)
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
    Expand ItemType(quantity>1) into a flat list of ItemType(quantity=1),
    assigning unique ids with #index suffix.
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


def _try_place_item_in_box(box: BoxInstance, item: ItemType, grid_resolution: float) -> bool:
    """
    Try to place item in box; return True if placed.
    """
    oris = orientations_of(item) if item.can_rotate else [(item.length, item.width, item.height)]

    # Prefer orientations that fill the box better
    def orientation_score(dims):
        l, w, h = dims
        bt = box.box_type
        fit = (l / bt.inner_length) * (w / bt.inner_width) * (h / bt.inner_height)
        return -fit

    oris.sort(key=orientation_score)

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
# 4. MULTI-SKU PACKER (LIST OF ITEMS)
# ============================================================

def pack_order(
    items: List[ItemType],
    box_types: List[BoxType],
    grid_resolution: float = 1.0,
) -> Tuple[List[BoxInstance], List[ItemType]]:
    """
    Pack a LIST of items (possibly multiple SKUs) into available box types.

    Logic:
    - Expand items and sort by volume (FFD).
    - For each item: try to place into existing boxes.
    - If not possible, open a new box chosen with a lookahead score that approximates
      "minimize number of boxes first", then "minimize waste", then "minimize cost".
    - Respect max_boxes per box type.

    Returns:
    - boxes: list of used BoxInstance
    - unassigned_items: list of items (quantity=1 each) that couldn't be packed
    """
    expanded = _expand_items(items)
    expanded.sort(key=lambda it: it.volume, reverse=True)

    # Sort box types by volume ascending for tie-breaks, but we evaluate all anyway.
    box_types_list = list(box_types)
    opened_counts: Dict[str, int] = {bt.id: 0 for bt in box_types_list}

    boxes: List[BoxInstance] = []
    unassigned: List[ItemType] = []

    def pile_volume_from(index: int) -> float:
        return sum(it.volume for it in expanded[index:])

    def can_open(bt: BoxType) -> bool:
        return (bt.max_boxes is None) or (opened_counts[bt.id] < bt.max_boxes)

    def open_new_box_for_item(item: ItemType, current_index: int) -> Optional[BoxInstance]:
        pile_vol = pile_volume_from(current_index)  # includes current item and remaining
        candidates: List[BoxType] = []

        for bt in box_types_list:
            if not can_open(bt):
                continue
            if _item_can_fit_in_box_type(item, bt):
                candidates.append(bt)

        if not candidates:
            return None

        def score(bt: BoxType):
            # Approximate "boxes needed" by volume (works well as a first-order proxy)
            approx_boxes_needed = ceil(pile_vol / bt.volume) if bt.volume > 0 else inf

            # Prefer smaller waste (if approx boxes equal)
            waste = (approx_boxes_needed * bt.volume) - pile_vol

            # Final tie-breaker: lower cost, then smaller box volume
            return (approx_boxes_needed, waste, bt.cost, bt.volume)

        candidates.sort(key=score)
        chosen = candidates[0]

        opened_counts[chosen.id] += 1
        idx = sum(1 for b in boxes if b.box_type.id == chosen.id) + 1
        new_box = BoxInstance(box_type=chosen, instance_index=idx)
        boxes.append(new_box)
        return new_box

    # Main loop
    for i, item in enumerate(expanded):
        placed = False

        # Try existing boxes first: prefer fuller ones
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
            # If we opened a box that geometrically fits the item but placement fails,
            # treat it as unassigned (rare with this heuristic).
            unassigned.append(item)

    return boxes, unassigned


# ============================================================
# 5. SINGLE-SKU PLANNER (PRECISE BOX-MIX LIKE YOUR SKU LOGIC)
# ============================================================

def _get_real_capacity_single_box(item: ItemType, box_type: BoxType, probe_resolution: float) -> int:
    """
    Estimate how many identical items fit in ONE box of box_type by simulation.
    This is far more reliable than pure tiling math for your SKU use-case.

    probe_resolution should be coarse for speed (e.g., min(item dims)).
    """
    if not _item_can_fit_in_box_type(item, box_type):
        return 0

    # Upper bound by volume (safe)
    max_theoretical = int(box_type.volume // item.volume) if item.volume > 0 else 0
    if max_theoretical <= 0:
        return 0

    # Build dummy items
    dummies = [ItemType("DUMMY", item.length, item.width, item.height, quantity=1, can_rotate=item.can_rotate)
               for _ in range(max_theoretical)]

    # Pack using ONLY this box type with max_boxes=1 (exactly one box)
    one_box_type = BoxType(
        id="PROBE",
        inner_length=box_type.inner_length,
        inner_width=box_type.inner_width,
        inner_height=box_type.inner_height,
        cost=0.0,
        max_boxes=1,
    )

    boxes, unassigned = pack_order(dummies, [one_box_type], grid_resolution=probe_resolution)
    if not boxes:
        return 0
    return len(boxes[0].items)


def _plan_mix_single_sku(
    quantity: int,
    box_types: List[BoxType],
    capacities: List[int],
) -> Dict[str, int]:
    """
    Find the best mix (fewest boxes, then cost, then waste volume).
    Uses DFS with pruning; works well when number of box types is small.
    """
    n = len(box_types)
    best_mix: Dict[str, int] = {bt.id: 0 for bt in box_types}
    best_score = (inf, inf, inf)  # (total_boxes, total_cost, total_volume)

    # Sort types by "bigger boxes first" to find low box-count solutions early
    order = sorted(range(n), key=lambda k: -box_types[k].volume)

    def evaluate(counts: List[int]) -> Tuple[float, float, float]:
        total_boxes = sum(counts)
        total_cost = sum(counts[i] * box_types[i].cost for i in range(n))
        total_vol = sum(counts[i] * box_types[i].volume for i in range(n))
        return (total_boxes, total_cost, total_vol)

    def dfs(pos: int, counts: List[int], covered: int, used_boxes: int):
        nonlocal best_mix, best_score

        # Prune: already worse in box-count
        if used_boxes > best_score[0]:
            return

        # If covered enough -> candidate solution
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
        # You never need more than ceil(remaining/cap) boxes of this type at this point
        needed = ceil((quantity - covered) / cap)
        limit = min(max_use, needed)

        # Try bigger counts first (find low box-count fast)
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
    1) Compute real capacity per box type (simulation in ONE box each, coarse resolution).
    2) Plan best mix: fewest boxes, then cost, then waste.
    3) Run pack_order once with those max_boxes limits.

    Returns:
    - boxes, unassigned, planned_box_types, mix
    """
    # Coarse probe resolution so capacity probing is fast even if grid_resolution=0.5
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
    Print a compact summary:
    - For each box: counts of each base item id.
    - Unassigned items (if any).
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
    Print how many boxes of each type were used and how many remain (based on max_boxes).
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
