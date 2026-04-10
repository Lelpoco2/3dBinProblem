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


def _run_benchmark(scenario: dict, repetitions: int = REPETITIONS) -> dict:
    if repetitions < 1:
        raise ValueError(f"repetitions must be >= 1, got {repetitions}")
    times_rounded = []
    result = None
    for _ in range(repetitions):
        t0 = time.perf_counter()
        result = _run_scenario_once(scenario)
        t1 = time.perf_counter()
        times_rounded.append(round(t1 - t0, 6))
    return {
        "name": scenario["name"],
        "mode": result["mode"],
        "repetitions": repetitions,
        "times_s": times_rounded,
        "min_s": min(times_rounded),
        "max_s": max(times_rounded),
        "avg_s": sum(times_rounded) / len(times_rounded),
        "total_boxes": result["total_boxes"],
        "box_breakdown": result["box_breakdown"],
        "unassigned_items": result["unassigned_items"],
    }


# ---------------------------------------------------------------------------
# Output — CSV
# ---------------------------------------------------------------------------

def _write_csv(results: list, timestamp: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    fieldnames = [
        "timestamp", "scenario", "mode", "repetitions",
        "min_s", "max_s", "avg_s",
        "total_boxes", "box_breakdown", "unassigned_items",
    ]
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in results:
            breakdown_str = ",".join(f"{k}:{v}" for k, v in sorted(r["box_breakdown"].items()))
            writer.writerow({
                "timestamp": timestamp,
                "scenario": r["name"],
                "mode": r["mode"],
                "repetitions": r["repetitions"],
                "min_s": r["min_s"],
                "max_s": r["max_s"],
                "avg_s": r["avg_s"],
                "total_boxes": r["total_boxes"],
                "box_breakdown": breakdown_str,
                "unassigned_items": r["unassigned_items"],
            })


# ---------------------------------------------------------------------------
# Output — JSON
# ---------------------------------------------------------------------------

def _write_json(results: list, timestamp: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "benchmark_results.json"
    # NOTE: read-then-write pattern has a narrow data-loss window on crash between
    # the two opens. Acceptable for a local benchmark tool.
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                all_runs = json.load(f)
            if not isinstance(all_runs, list):
                all_runs = []
        except json.JSONDecodeError:
            all_runs = []  # file corrotto: si riparte da zero
    else:
        all_runs = []
    all_runs.append({"run_timestamp": timestamp, "scenarios": results})
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)


# ---------------------------------------------------------------------------
# Output — console
# ---------------------------------------------------------------------------

def _print_results(results: list, box_ids: list, timestamp: str) -> None:
    box_headers = "  ".join(f"{bid:>7}" for bid in box_ids)
    header = (
        f"{'Scenario':<22} | {'Mode':<10} | {'Reps':>4} | "
        f"{'Min(s)':>8} | {'Max(s)':>8} | {'Avg(s)':>8} | "
        f"{'Boxes':>5}  {box_headers}  {'Unassigned':>10}"
    )
    sep = "-" * len(header)
    print(f"\n=== 3D Bin Packing Benchmark — {timestamp} ===\n")
    print(header)
    print(sep)
    for r in results:
        box_counts = "  ".join(
            f"{r['box_breakdown'].get(bid, 0):>7}" for bid in box_ids
        )
        print(
            f"{r['name']:<22} | {r['mode']:<10} | {r['repetitions']:>4} | "
            f"{r['min_s']:>8.4f} | {r['max_s']:>8.4f} | {r['avg_s']:>8.4f} | "
            f"{r['total_boxes']:>5}  {box_counts}  {r['unassigned_items']:>10}"
        )
    print(f"\nResults saved to: {RESULTS_DIR}/")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name": "single_sku_tiny",
        "items": [ItemType("SKU_A", 5, 4, 3, quantity=3, name="Gadget-S")],
        "box_types": _BOXES_SM,
    },
    {
        "name": "single_sku_medium",
        "items": [ItemType("SKU_A", 5, 4, 3, quantity=20, name="Gadget-S")],
        "box_types": _BOXES_SML,
    },
    {
        "name": "multi_sku_small",
        "items": [
            ItemType("SKU_A", 5, 4, 3, quantity=2, name="Gadget-A"),
            ItemType("SKU_B", 8, 6, 4, quantity=2, name="Gadget-B"),
            ItemType("SKU_C", 12, 10, 8, quantity=2, name="Gadget-C"),
            ItemType("SKU_D", 6, 5, 4, quantity=2, name="Gadget-D"),
        ],
        "box_types": _BOXES_SML,
    },
    {
        "name": "multi_sku_stress",
        "items": [
            ItemType("SKU_A", 5, 4, 3, quantity=5, name="Gadget-A"),
            ItemType("SKU_B", 8, 6, 4, quantity=4, name="Gadget-B"),
            ItemType("SKU_C", 12, 10, 8, quantity=3, name="Gadget-C"),
            ItemType("SKU_D", 6, 5, 4, quantity=4, name="Gadget-D"),
            ItemType("SKU_E", 9, 7, 5, quantity=3, name="Gadget-E"),
            ItemType("SKU_F", 14, 11, 9, quantity=3, name="Gadget-F"),
            ItemType("SKU_G", 4, 3, 2, quantity=5, name="Gadget-G"),
            ItemType("SKU_H", 7, 6, 3, quantity=4, name="Gadget-H"),
        ],
        "box_types": _BOXES_SML,
    },
    {
        "name": "single_sku_large",
        "items": [ItemType("SKU_A", 5, 4, 3, quantity=50, name="Gadget-S")],
        "box_types": _BOXES_SML,
    },
    {
        "name": "single_sku_100",
        "items": [ItemType("SKU_A", 5, 4, 3, quantity=100, name="Gadget-S")],
        "box_types": _BOXES_SML,
    },
    {
        "name": "multi_sku_100",
        "items": [
            ItemType("SKU_A", 5, 4, 3, quantity=10, name="Item-A"),
            ItemType("SKU_B", 8, 6, 4, quantity=10, name="Item-B"),
            ItemType("SKU_C", 6, 5, 4, quantity=10, name="Item-C"),
            ItemType("SKU_D", 4, 3, 2, quantity=10, name="Item-D"),
            ItemType("SKU_E", 9, 7, 5, quantity=10, name="Item-E"),
            ItemType("SKU_F", 7, 6, 3, quantity=10, name="Item-F"),
            ItemType("SKU_G", 3, 3, 3, quantity=10, name="Item-G"),
            ItemType("SKU_H", 10, 8, 6, quantity=10, name="Item-H"),
            ItemType("SKU_I", 5, 5, 5, quantity=10, name="Item-I"),
            ItemType("SKU_J", 6, 4, 3, quantity=10, name="Item-J"),
        ],
        "box_types": _BOXES_SML,
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print("Running benchmarks, please wait...")
    results = [_run_benchmark(s) for s in SCENARIOS]

    # Collect unique box IDs preserving insertion order
    seen: dict = {}
    for s in SCENARIOS:
        for bt in s["box_types"]:
            seen[bt.id] = True
    box_ids = list(seen.keys())

    _print_results(results, box_ids, timestamp)
    _write_csv(results, timestamp)
    _write_json(results, timestamp)


if __name__ == "__main__":
    main()
