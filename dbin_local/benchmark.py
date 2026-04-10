# dbin_local/benchmark.py
"""
Standalone benchmark for the 3D bin-packing algorithm.
Measures execution speed across 7 scenarios of increasing complexity.

Run from dbin_local/:  python benchmark.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from BinCore import BoxType, ItemType, pack_order, pack_single_sku_order

REPETITIONS = 5
RESULTS_DIR = Path(__file__).parent / "benchmark_results"

_BOX_S = BoxType(id="BOX_S", name="Pacco-S", inner_length=15, inner_width=12, inner_height=10, cost=1.0)
_BOX_M = BoxType(id="BOX_M", name="Pacco-M", inner_length=25, inner_width=20, inner_height=15, cost=2.0)
_BOX_L = BoxType(id="BOX_L", name="Pacco-L", inner_length=35, inner_width=28, inner_height=20, cost=3.5)

_BOXES_SM  = [_BOX_S, _BOX_M]
_BOXES_SML = [_BOX_S, _BOX_M, _BOX_L]


# ---------------------------------------------------------------------------
# Smart pack routing (mirrors demo_ultimate.py logic)
# ---------------------------------------------------------------------------

def _smart_pack(items, box_types, grid_resolution=1.0):
    if len(items) == 1:
        boxes, unassigned, planned_box_types, _ = pack_single_sku_order(
            items[0], box_types, grid_resolution=grid_resolution
        )
        return boxes, unassigned, planned_box_types
    boxes, unassigned = pack_order(items, box_types, grid_resolution=grid_resolution)
    return boxes, unassigned, box_types


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def _run_scenario_once(scenario: dict) -> dict:
    items = scenario["items"]
    box_types = scenario["box_types"]
    mode = "single-SKU" if len(items) == 1 else "multi-SKU"
    boxes, unassigned, _ = _smart_pack(items, box_types)
    box_ids = [bt.id for bt in box_types]
    breakdown = Counter(b.box_type.id for b in boxes)
    box_breakdown = {bid: breakdown.get(bid, 0) for bid in box_ids}
    return {
        "mode": mode,
        "total_boxes": len(boxes),
        "box_breakdown": box_breakdown,
        "unassigned_items": len(unassigned),
    }
