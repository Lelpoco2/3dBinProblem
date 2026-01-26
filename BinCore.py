"""
3D bin-packing / box recommendation algorithm (geometry-only).

Heuristic:
- First-Fit-Decreasing by item volume.
- Simple 3D placement with a bottom-left-back scan.
- When opening a new box, choose the box type that minimizes
  number of boxes needed (via capacity estimate), with tie-breaking
  by higher capacity.

Goal: minimize number of boxes first, then minimize number of box types.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import itertools
from collections import Counter
from math import inf

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
        """Volume of a single item."""
        return self.length * self.width * self.height


@dataclass
class BoxType:
    """
    Represents a type of box/bag that can be used.

    max_boxes:
        Maximum number of boxes of this type that can be used.
        If None, there is no limit.

    cost:
        Economic cost of using one box of this type. This can be:
        - packaging material cost,
        - estimated shipping cost,
        - or any combined score you choose.
    """
    id: str
    inner_length: float
    inner_width: float
    inner_height: float
    cost: float
    max_boxes: Optional[int] = None

    @property
    def volume(self) -> float:
        """Internal volume of the box."""
        return self.inner_length * self.inner_width * self.inner_height


@dataclass
class PlacedItem:
    """
    Represents one concrete item instance placed in a specific box.
    Geometry-only; no weight tracking.
    """
    item_id: str
    box_id: str
    length: float
    width: float
    height: float
    position: Tuple[float, float, float]  # (x, y, z) corner
    rotation: Tuple[float, float, float]  # (L, W, H) after rotation


@dataclass
class BoxInstance:
    """
    Represents a concrete box that we are filling with items.
    """
    box_type: BoxType
    instance_index: int
    items: List[PlacedItem] = field(default_factory=list)

    def used_volume(self) -> float:
        """Total volume of items placed in this box."""
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
    Check if an item with rotated dimensions 'rot_dims' can fit
    inside the internal dimensions of 'box'.
    """
    l, w, h = rot_dims
    return (
        l <= box.inner_length and
        w <= box.inner_width and
        h <= box.inner_height
    )


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
    dims: Tuple[float, float, float]
) -> bool:
    """
    Check if placing an item at 'position' with 'dims' would collide
    with any already placed item in 'box'.
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
    Range for floats, inclusive of 'stop' (with small epsilon).
    """
    v = start
    while v <= stop + 1e-9:
        yield v
        v += step


def find_feasible_position(
    box: BoxInstance,
    dims: Tuple[float, float, float],
    resolution: float = 1.0,
) -> Optional[Tuple[float, float, float]]:
    """
    Find a feasible position for an item within 'box' using a simple
    'bottom-left-back' scanning heuristic.

    Scans positions starting from (0,0,0) and also tries key positions
    adjacent to already placed items for better packing.
    """
    l, w, h = dims
    L = box.box_type.inner_length
    W = box.box_type.inner_width
    H = box.box_type.inner_height

    if l > L or w > W or h > H:
        return None

    # Generate candidate positions
    candidate_positions = set()

    # Always try origin
    candidate_positions.add((0.0, 0.0, 0.0))

    # Add positions adjacent to existing items
    for placed in box.items:
        px, py, pz = placed.position
        pl, pw, ph = placed.rotation

        adjacent_positions = [
            (px + pl, py, pz),      # right of item
            (px, py + pw, pz),      # behind item
            (px, py, pz + ph),      # on top of item
            (px - l, py, pz),       # left of item
            (px, py - w, pz),       # in front of item
            (px, py, pz - h),       # below item
        ]

        for pos in adjacent_positions:
            x, y, z = pos
            if (
                0 <= x <= L - l + 1e-6 and
                0 <= y <= W - w + 1e-6 and
                0 <= z <= H - h + 1e-6
            ):
                candidate_positions.add((round(x, 6), round(y, 6), round(z, 6)))

    # Also add grid positions
    xs = [round(v, 6) for v in frange(0.0, L - l + 1e-6, resolution)]
    ys = [round(v, 6) for v in frange(0.0, W - w + 1e-6, resolution)]
    zs = [round(v, 6) for v in frange(0.0, H - h + 1e-6, resolution)]

    for z in zs:
        for y in ys:
            for x in xs:
                candidate_positions.add((x, y, z))

    # Sort positions by (z, y, x) for bottom-left-back priority
    sorted_positions = sorted(candidate_positions, key=lambda p: (p[2], p[1], p[0]))

    # Try each position
    for pos in sorted_positions:
        if not collides_with_existing(box, pos, dims):
            return pos

    return None


# ============================================================
# 2b. CAPACITY ESTIMATOR (for box selection and mix planning)
# ============================================================

def estimate_copies_in_box(item: ItemType, box: BoxType) -> int:
    """
    Estimate how many copies of 'item' can fit into an empty 'box'
    using simple axis-aligned tiling (no complex packing).

    Used to choose best box type and to plan box mixes.
    """
    if item.can_rotate:
        oris = orientations_of(item)
    else:
        oris = [(item.length, item.width, item.height)]

    best = 0
    for (l, w, h) in oris:
        if l > box.inner_length or w > box.inner_width or h > box.inner_height:
            continue
        nx = int(box.inner_length // l)
        ny = int(box.inner_width // w)
        nz = int(box.inner_height // h)
        best = max(best, nx * ny * nz)

    return best


# ============================================================
# 3. CORE PACKING ALGORITHM (OPTIMIZED FOR FEWER BOXES)
# ============================================================

def pack_order(
    items: List[ItemType],
    box_types: List[BoxType],
    grid_resolution: float = 1.0,
) -> Tuple[List[BoxInstance], List[ItemType]]:
    """
    Pack all items into boxes using a First-Fit-Decreasing heuristic,
    respecting per-type box limits (max_boxes).

    When opening a new box for a given item, choose the box type
    that minimizes the number of boxes needed for remaining items
    (via capacity estimate), preferring higher capacity as tie-breaker.

    Returns
    -------
    boxes : list[BoxInstance]
        All boxes used and the items placed in each.
    unassigned_items : list[ItemType]
        Items that could not be packed in any box type.
    """
    # 1) Expand items by quantity
    expanded_items: List[ItemType] = []
    for it in items:
        for i in range(it.quantity):
            expanded_items.append(
                ItemType(
                    id=f"{it.id}#{i+1}",
                    length=it.length,
                    width=it.width,
                    height=it.height,
                    quantity=1,
                    can_rotate=it.can_rotate,
                )
            )

    # 2) Sort items (First-Fit-Decreasing by volume)
    expanded_items.sort(key=lambda it: it.volume, reverse=True)

    # 3) Sort box types by volume (larger boxes first)
    box_types_sorted = sorted(box_types, key=lambda bt: -bt.volume)

    boxes: List[BoxInstance] = []
    unassigned_items: List[ItemType] = []

    # Track how many boxes of each type we have opened (for max_boxes)
    opened_counts = {bt.id: 0 for bt in box_types_sorted}

    # Helper: how many items of the same base id remain from current_index onward
    def remaining_same_type(current_index: int, base_id: str) -> int:
        count = 0
        for it2 in expanded_items[current_index:]:
            if it2.id.split("#")[0] == base_id:
                count += 1
        return count

    # Helper: try placing in a given box
    def try_place_in_box(box: BoxInstance, item: ItemType) -> bool:
        if item.can_rotate:
            orientations = orientations_of(item)
        else:
            orientations = [(item.length, item.width, item.height)]

        def orientation_score(dims):
            l, w, h = dims
            bt = box.box_type
            fit_score = (l / bt.inner_length) * (w / bt.inner_width) * (h / bt.inner_height)
            return -fit_score  # prefer better fit (higher product)

        orientations.sort(key=orientation_score)

        for dims in orientations:
            if not fits_in_box(dims, box.box_type):
                continue
            pos = find_feasible_position(box, dims, resolution=grid_resolution)
            if pos is not None:
                placed = PlacedItem(
                    item_id=item.id,
                    box_id=f"{box.box_type.id}-{box.instance_index}",
                    length=item.length,
                    width=item.width,
                    height=item.height,
                    position=pos,
                    rotation=dims,
                )
                box.items.append(placed)
                return True

        return False

    # Helper: check if an item can physically fit in a box type
    def item_can_fit_in_box_type(item: ItemType, bt: BoxType) -> bool:
        """Check if item can physically fit in box type (ignoring stock limits)."""
        if item.can_rotate:
            oris = orientations_of(item)
        else:
            oris = [(item.length, item.width, item.height)]
        return any(fits_in_box(o, bt) for o in oris)

    # Helper: open a new box for an item, respecting max_boxes
    def open_new_box_for_item(item: ItemType, current_index: int) -> Optional[BoxInstance]:
        """
        Choose the best box type that can hold 'item', prioritizing:
        1. Fewer boxes needed for remaining items (via capacity)
        2. Higher capacity as tie-breaker
        """
        candidate_types = []
        for bt in box_types_sorted:
            # Skip if we've reached stock limit for this box type
            if bt.max_boxes is not None and opened_counts[bt.id] >= bt.max_boxes:
                continue
            if item_can_fit_in_box_type(item, bt):
                candidate_types.append(bt)

        if not candidate_types:
            # Check if item can fit in ANY box type (ignoring stock)
            can_fit_anywhere = any(item_can_fit_in_box_type(item, bt) for bt in box_types_sorted)
            if can_fit_anywhere:
                # Item could fit but we're out of stock
                return None
            else:
                # Item is too big for any box type
                return None

        base_id = item.id.split("#")[0]
        remaining = remaining_same_type(current_index, base_id)

        def score(bt: BoxType):
            capacity = estimate_copies_in_box(item, bt)
            if capacity == 0:
                return (float("inf"), 0)
            boxes_needed = (remaining + capacity - 1) // capacity  # ceil division
            # Prefer fewer boxes needed, then higher capacity
            return (boxes_needed, -capacity)

        candidate_types.sort(key=score)
        chosen = candidate_types[0]

        opened_counts[chosen.id] += 1
        idx = sum(1 for b in boxes if b.box_type.id == chosen.id) + 1
        new_box = BoxInstance(box_type=chosen, instance_index=idx)
        boxes.append(new_box)
        return new_box

    # 4) Main packing loop
    for idx, item in enumerate(expanded_items):
        placed = False

        # Try existing boxes (sorted by volume utilization to prefer fuller boxes)
        boxes_sorted = sorted(
            boxes,
            key=lambda b: -(
                b.used_volume() / b.box_type.volume if b.box_type.volume > 0 else 0
            ),
        )

        for box in boxes_sorted:
            if try_place_in_box(box, item):
                placed = True
                break

        if placed:
            continue

        # Open new box type if possible (within limits)
        new_box = open_new_box_for_item(item, idx)
        if new_box is None:
            unassigned_items.append(item)
            continue

        if not try_place_in_box(new_box, item):
            unassigned_items.append(item)

    return boxes, unassigned_items


# ============================================================
# 3b. EXHAUSTIVE BOX MIX FOR SINGLE SKU
# ============================================================

def exhaustive_box_mix_for_single_sku(
    item: ItemType,
    box_types: List[BoxType],
    grid_resolution: float = 1.0,
) -> Tuple[List[BoxInstance], List[BoxType], dict[str, int]]:
    """
    Exhaustively enumerate all feasible combinations of box counts
    (within max_boxes) for a single SKU, run the 3D packer for each,
    and select the combination that:

    1) Minimizes total number of boxes,
    2) Minimizes total box cost,
    3) Packs the maximum number of items,
    4) Minimizes number of distinct box types.
    """
    from itertools import product

    Q = item.quantity

    # Range for each box type: 0 .. max_boxes (or 0..Q if None)
    ranges = []
    for bt in box_types:
        upper = bt.max_boxes if bt.max_boxes is not None else Q
        ranges.append(range(0, upper + 1))

    best_total_boxes = inf
    best_cost = inf
    best_packed = -1
    best_distinct_types = inf

    best_boxes: List[BoxInstance] = []
    best_box_types: List[BoxType] = []
    best_mix: dict[str, int] = {bt.id: 0 for bt in box_types}

    for counts in product(*ranges):
        if sum(counts) == 0:
            continue  # skip trivial "no boxes" combo

        # Build planned box types according to counts
        planned_box_types: List[BoxType] = []
        for bt, cnt in zip(box_types, counts):
            planned_box_types.append(
                BoxType(
                    id=bt.id,
                    inner_length=bt.inner_length,
                    inner_width=bt.inner_width,
                    inner_height=bt.inner_height,
                    cost=bt.cost,
                    max_boxes=cnt,
                )
            )

        boxes, unassigned = pack_order([item], planned_box_types, grid_resolution=grid_resolution)
        packed = Q - len(unassigned)
        total_boxes = len(boxes)
        cost = sum(b.box_type.cost for b in boxes)
        distinct_types = sum(1 for c in counts if c > 0)

        better = False
        if total_boxes < best_total_boxes:
            better = True
        elif total_boxes == best_total_boxes:
            if cost < best_cost:
                better = True
            elif cost == best_cost:
                if packed > best_packed:
                    better = True
                elif packed == best_packed and distinct_types < best_distinct_types:
                    better = True

        if better:
            best_total_boxes = total_boxes
            best_cost = cost
            best_packed = packed
            best_distinct_types = distinct_types
            best_boxes = boxes
            best_box_types = planned_box_types
            best_mix = {bt.id: c for bt, c in zip(box_types, counts)}

    return best_boxes, best_box_types, best_mix


def plan_box_mix_for_single_sku(
    item: ItemType,
    box_types: List[BoxType],
    quantity: int,
) -> dict[str, int]:
    """
    For a single SKU 'item' and a list of BoxType, compute how many boxes
    of each type to use to cover 'quantity' units.

    Priority:
    - minimize total number of boxes
    - among equal box counts, prefer fewer distinct box types

    Cost is ignored here.
    """
    capacities = [estimate_copies_in_box(item, bt) for bt in box_types]
    if all(cap == 0 for cap in capacities):
        return {bt.id: 0 for bt in box_types}

    best_mix = {bt.id: 0 for bt in box_types}
    best_score = (float("inf"), float("inf"))  # (total_boxes, distinct_types)

    def evaluate_mix(mix: dict[str, int]) -> tuple[int, int]:
        total_boxes = sum(mix.values())
        distinct_types = sum(1 for v in mix.values() if v > 0)
        return (total_boxes, distinct_types)

    def backtrack(index: int, current_counts: List[int], covered: int):
        nonlocal best_mix, best_score

        if covered >= quantity:
            candidate_mix = {bt.id: c for bt, c in zip(box_types, current_counts)}
            cand_score = evaluate_mix(candidate_mix)
            if cand_score < best_score:
                best_score = cand_score
                best_mix = candidate_mix
            return

        if index == len(box_types):
            return

        # crude upper bound pruning
        remaining_potential = covered
        for j in range(index, len(box_types)):
            cap_j = capacities[j]
            bt_j = box_types[j]
            max_use_j = bt_j.max_boxes if bt_j.max_boxes is not None else quantity
            remaining_potential += cap_j * max_use_j
        if remaining_potential < quantity:
            return

        bt = box_types[index]
        cap = capacities[index]
        max_use = bt.max_boxes if bt.max_boxes is not None else quantity

        for cnt in range(0, max_use + 1):
            added_cover = cap * cnt
            new_covered = covered + added_cover
            current_counts[index] = cnt
            backtrack(index + 1, current_counts, new_covered)

        current_counts[index] = 0

    backtrack(0, [0] * len(box_types), covered=0)
    return best_mix


def pack_single_sku_order(
    item: ItemType,
    box_types: List[BoxType],
    grid_resolution: float = 1.0,
) -> Tuple[List[BoxInstance], List[ItemType], List[BoxType], dict[str, int]]:
    """
    Single-SKU flow:

    - plan box mix prioritizing fewer boxes (then fewer box types),
    - clamp counts by stock,
    - call pack_order once with those max_boxes.
    """
    total_qty = item.quantity
    mix = plan_box_mix_for_single_sku(item, box_types, total_qty)

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
# 4. SUMMARY PRINTING
# ============================================================

def print_packing_summary(boxes: List[BoxInstance], unassigned_items: List[ItemType]) -> None:
    """
    Print a compact summary:
    - For each box: how many of each item type.
    - Then list of unassigned item types (if any).
    """
    for b in boxes:
        counter = Counter(it.item_id.split("#")[0] for it in b.items)
        vol_util = (b.used_volume() / b.box_type.volume * 100) if b.box_type.volume > 0 else 0
        print(f"Box {b.box_type.id}-{b.instance_index}")
        print(
            f" Box size: "
            f"{b.box_type.inner_length}x{b.box_type.inner_width}x{b.box_type.inner_height}"
        )
        print(f" Volume utilization: {vol_util:.1f}%")
        print(" Items:")
        for item_id, qty in counter.items():
            print(f"  - {item_id}: {qty}")
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
    Print how many boxes of each type were used and how many remain
    (based on max_boxes).
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