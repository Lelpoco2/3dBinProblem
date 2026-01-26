"""
3D bin-packing / box recommendation algorithm (geometry-only).

Heuristic:
- First-Fit-Decreasing by item volume.
- Bottom-left-back scan (Corner Points + Optional Grid).
- Smart "Real Capacity" pre-check with dynamic resolution for speed.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import itertools
from collections import Counter
from math import inf

# ============================================================
# 1. DATA MODELS
# ============================================================

@dataclass
class ItemType:
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
    item_id: str
    box_id: str
    length: float
    width: float
    height: float
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]


@dataclass
class BoxInstance:
    box_type: BoxType
    instance_index: int
    items: List[PlacedItem] = field(default_factory=list)

    def used_volume(self) -> float:
        return sum(it.rotation[0] * it.rotation[1] * it.rotation[2] for it in self.items)


# ============================================================
# 2. GEOMETRY HELPERS
# ============================================================

def orientations_of(item: ItemType) -> List[Tuple[float, float, float]]:
    dims = (item.length, item.width, item.height)
    return list(set(itertools.permutations(dims, 3)))


def fits_in_box(rot_dims: Tuple[float, float, float], box: BoxType) -> bool:
    l, w, h = rot_dims
    return l <= box.inner_length and w <= box.inner_width and h <= box.inner_height


def aabb_overlap(a_min, a_max, b_min, b_max) -> bool:
    # Standard AABB overlap check
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

    # Check against all placed items
    for placed in box.items:
        px, py, pz = placed.position
        pl, pw, ph = placed.rotation
        old_min = (px, py, pz)
        old_max = (px + pl, py + pw, pz + ph)
        if aabb_overlap(new_min, new_max, old_min, old_max):
            return True
    return False


def frange(start: float, stop: float, step: float):
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
    Scans for a valid position.
    Optimization: Only uses the fine 'resolution' grid if absolutely necessary
    and if the number of points is manageable.
    """
    l, w, h = dims
    L = box.box_type.inner_length
    W = box.box_type.inner_width
    H = box.box_type.inner_height

    if l > L or w > W or h > H:
        return None

    # 1. Always check corner points (Origin + Adjacent to existing items)
    candidate_positions = set()
    candidate_positions.add((0.0, 0.0, 0.0))

    for placed in box.items:
        px, py, pz = placed.position
        pl, pw, ph = placed.rotation
        # Try 6 adjacent spots
        adj = [
            (px + pl, py, pz), (px, py + pw, pz), (px, py, pz + ph),
            (px - l, py, pz),  (px, py - w, pz),  (px, py, pz - h)
        ]
        for (x, y, z) in adj:
            if 0 <= x <= L - l + 1e-6 and 0 <= y <= W - w + 1e-6 and 0 <= z <= H - h + 1e-6:
                candidate_positions.add((round(x, 6), round(y, 6), round(z, 6)))

    # Sort 'Corner' candidates first (Bottom-Back-Left)
    sorted_candidates = sorted(candidate_positions, key=lambda p: (p[2], p[1], p[0]))
    for pos in sorted_candidates:
        if not collides_with_existing(box, pos, dims):
            return pos

    # 2. Fallback: Grid Search (Only if corners failed)
    # OPTIMIZATION: If resolution is too fine, this loop kills performance.
    # We only run it if the total point count is reasonable (< 5000 checks).
    
    # Estimate grid steps
    nx = int((L - l) // resolution) + 1
    ny = int((W - w) // resolution) + 1
    nz = int((H - h) // resolution) + 1
    
    total_points = nx * ny * nz
    
    # If too many points, force a coarser resolution dynamically for this check
    if total_points > 5000:
        # Scale resolution up to keep points under ~2000
        factor = (total_points / 2000) ** (1/3)
        resolution *= factor
    
    xs = [round(v, 6) for v in frange(0.0, L - l + 1e-6, resolution)]
    ys = [round(v, 6) for v in frange(0.0, W - w + 1e-6, resolution)]
    zs = [round(v, 6) for v in frange(0.0, H - h + 1e-6, resolution)]

    # We iterate Z, then Y, then X
    for z in zs:
        for y in ys:
            for x in xs:
                # Skip if we already checked this in step 1
                if (x, y, z) in candidate_positions:
                    continue
                if not collides_with_existing(box, (x,y,z), dims):
                    return (x,y,z)

    return None


# ============================================================
# 3. CORE PACKING
# ============================================================

def pack_order(
    items: List[ItemType],
    box_types: List[BoxType],
    grid_resolution: float = 1.0,
) -> Tuple[List[BoxInstance], List[ItemType]]:
    
    # Expand
    expanded_items: List[ItemType] = []
    for it in items:
        for i in range(it.quantity):
            expanded_items.append(ItemType(
                id=f"{it.id}#{i+1}",
                length=it.length, width=it.width, height=it.height,
                quantity=1, can_rotate=it.can_rotate
            ))

    # Sort items desc volume
    expanded_items.sort(key=lambda it: it.volume, reverse=True)
    
    # Sort boxes desc volume (for iteration order)
    box_types_sorted = sorted(box_types, key=lambda bt: -bt.volume)

    boxes: List[BoxInstance] = []
    unassigned_items: List[ItemType] = []
    opened_counts: Dict[str, int] = {bt.id: 0 for bt in box_types_sorted}

    def try_place_in_box(box: BoxInstance, item: ItemType) -> bool:
        oris = orientations_of(item) if item.can_rotate else [(item.length, item.width, item.height)]
        # Heuristic: maximize fit score
        oris.sort(key=lambda d: -((d[0]/box.box_type.inner_length)*(d[1]/box.box_type.inner_width)*(d[2]/box.box_type.inner_height)))
        
        for dims in oris:
            if not fits_in_box(dims, box.box_type): continue
            pos = find_feasible_position(box, dims, resolution=grid_resolution)
            if pos:
                box.items.append(PlacedItem(
                    item_id=item.id, box_id=f"{box.box_type.id}-{box.instance_index}",
                    length=item.length, width=item.width, height=item.height,
                    position=pos, rotation=dims
                ))
                return True
        return False

    def open_new_box_for_item(item: ItemType) -> Optional[BoxInstance]:
        # Simple greedy selection used ONLY during the "final pack" phase
        # The smart selection happens in the 'Single SKU Planner' before this.
        candidates = [bt for bt in box_types_sorted 
                      if (bt.max_boxes is None or opened_counts[bt.id] < bt.max_boxes)
                      and any(fits_in_box(o, bt) for o in (orientations_of(item) if item.can_rotate else [(item.length, item.width, item.height)]))]
        
        if not candidates: return None
        # Prefer smallest volume box that fits the item
        candidates.sort(key=lambda b: b.volume)
        chosen = candidates[0]
        
        opened_counts[chosen.id] += 1
        idx = sum(1 for b in boxes if b.box_type.id == chosen.id) + 1
        new_box = BoxInstance(box_type=chosen, instance_index=idx)
        boxes.append(new_box)
        return new_box

    for item in expanded_items:
        placed = False
        # Try existing boxes
        for box in sorted(boxes, key=lambda b: -b.used_volume()):
            if try_place_in_box(box, item):
                placed = True; break
        
        if not placed:
            new_box = open_new_box_for_item(item)
            if new_box and try_place_in_box(new_box, item):
                continue
            unassigned_items.append(item)

    return boxes, unassigned_items


# ============================================================
# 4. FAST & ACCURATE PLANNER
# ============================================================

def get_real_capacity(item: ItemType, box_type: BoxType) -> int:
    """
    Calculates actual box capacity.
    SPEED FIX: Uses the ITEM's dimensions as the grid resolution.
    """
    max_theoretical = int(box_type.volume // item.volume)
    if max_theoretical == 0: return 0
    
    # Optimization: If max_theoretical is huge, cap it? 
    # Actually, the grid speedup below makes this fast enough.
    
    dummy_items = [ItemType("test", item.length, item.width, item.height, 1, item.can_rotate) for _ in range(max_theoretical)]
    test_box_type = BoxType("test", box_type.inner_length, box_type.inner_width, box_type.inner_height, 0, max_boxes=1)
    
    # KEY PERFORMANCE FIX: Use item min dimension as resolution
    # This avoids checking 0.5mm increments when the item is 10mm big.
    smart_resolution = min(item.length, item.width, item.height)
    
    boxes, _ = pack_order(dummy_items, [test_box_type], grid_resolution=smart_resolution)
    return len(boxes[0].items) if boxes else 0


def plan_mix_fast(item: ItemType, box_types: List[BoxType], quantity: int, capacities: List[int]) -> dict:
    best_mix = {bt.id: 0 for bt in box_types}
    best_score = (inf, inf, inf) # (boxes, cost, volume)

    def evaluate(mix):
        return (sum(mix.values()), sum(mix[bt.id]*bt.cost for bt in box_types), sum(mix[bt.id]*bt.volume for bt in box_types))

    def solve(idx, current_counts, covered):
        nonlocal best_mix, best_score
        if covered >= quantity:
            score = evaluate({bt.id: c for bt, c in zip(box_types, current_counts)})
            if score < best_score:
                best_score = score
                best_mix = {bt.id: c for bt, c in zip(box_types, current_counts)}
            return

        if idx == len(box_types): return

        # Pruning
        max_possible = covered
        for j in range(idx, len(box_types)):
            cap = capacities[j]
            limit = box_types[j].max_boxes if box_types[j].max_boxes is not None else quantity
            max_possible += cap * limit
        if max_possible < quantity: return

        # Try loop
        bt = box_types[idx]
        cap = capacities[idx]
        limit = bt.max_boxes if bt.max_boxes is not None else quantity
        
        # Heuristic optimization: try Max first? No, standard 0..limit is fine for small N
        for c in range(limit + 1):
            current_counts[idx] = c
            solve(idx + 1, current_counts, covered + cap * c)
        current_counts[idx] = 0

    solve(0, [0]*len(box_types), 0)
    return best_mix


def pack_single_sku_order(
    item: ItemType,
    box_types: List[BoxType],
    grid_resolution: float = 1.0,
) -> Tuple[List[BoxInstance], List[ItemType], List[BoxType], dict[str, int]]:
    
    # 1. Pre-calculate capacities (FAST)
    caps = [get_real_capacity(item, bt) for bt in box_types]
    
    # 2. Plan best mix
    mix = plan_mix_fast(item, box_types, item.quantity, caps)
    
    # 3. Final Pack
    planned_types = []
    for bt in box_types:
        limit = mix.get(bt.id, 0)
        if bt.max_boxes is not None: limit = min(limit, bt.max_boxes)
        planned_types.append(BoxType(bt.id, bt.inner_length, bt.inner_width, bt.inner_height, bt.cost, limit))
        
    boxes, unassigned = pack_order([item], planned_types, grid_resolution=grid_resolution)
    return boxes, unassigned, planned_types, mix

# ============================================================
# 5. PRINTING
# ============================================================
def print_packing_summary(boxes: List[BoxInstance], unassigned_items: List[ItemType]) -> None:
    for b in boxes:
        c = Counter(it.item_id.split("#")[0] for it in b.items)
        util = (b.used_volume()/b.box_type.volume*100) if b.box_type.volume>0 else 0
        print(f"Box {b.box_type.id}-{b.instance_index} | Size: {b.box_type.inner_length}x{b.box_type.inner_width}x{b.box_type.inner_height} | Util: {util:.1f}%")
        for i, q in c.items(): print(f" - {i}: {q}")
        print()
    if unassigned_items: print(f"Unpacked: {len(unassigned_items)}")
    else: print("All packed.")

def print_box_stock_usage(box_types: List[BoxType], boxes: List[BoxInstance]) -> None:
    used = Counter(b.box_type.id for b in boxes)
    print("Stock Usage:")
    for bt in box_types:
        u = used.get(bt.id, 0)
        rem = "inf" if bt.max_boxes is None else max(bt.max_boxes - u, 0)
        print(f" - {bt.id}: {u} used, {rem} remaining")
    print()
