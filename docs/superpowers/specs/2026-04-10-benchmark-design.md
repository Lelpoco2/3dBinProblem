# Benchmark System Design — 3D Bin Packing
**Date:** 2026-04-10

## Overview

A standalone benchmark script (`dbin_local/benchmark.py`) that measures execution speed
of the packing algorithm across 7 scenarios of increasing complexity. Results are printed
to console and saved persistently as both CSV and JSON.

---

## File

| Path | Description |
|------|-------------|
| `dbin_local/benchmark.py` | Single script — scenarios, runner, output |
| `dbin_local/benchmark_results/benchmark_results.csv` | Appended on each run |
| `dbin_local/benchmark_results/benchmark_results.json` | Appended on each run |

The `benchmark_results/` directory is created automatically if it does not exist.

---

## Logic

The script imports `BinCore` (same as `demo_ultimate.py`) and replicates the `smart_pack`
routing function locally (no import from `demo_ultimate` to avoid triggering its
`__main__` block):

```
if len(items) == 1 SKU type → pack_single_sku_order
else                        → pack_order
```

Each scenario is run **N = 5 repetitions** using `time.perf_counter` (not `timeit.timeit`,
to keep the measured result accessible). Min, max, and average times are recorded.
The box breakdown (count per box type) is taken from the **last** repetition result.

---

## Scenarios

| # | Name | Mode | SKUs | Total items | Box types |
|---|------|------|------|-------------|-----------|
| 1 | `single_sku_tiny` | single-SKU | 1 | 3 | BOX_S, BOX_M |
| 2 | `single_sku_medium` | single-SKU | 1 | 20 | BOX_S, BOX_M, BOX_L |
| 3 | `multi_sku_small` | multi-SKU | 4 | ~8 | BOX_S, BOX_M, BOX_L |
| 4 | `multi_sku_stress` | multi-SKU | 8 | ~30 | BOX_S, BOX_M, BOX_L |
| 5 | `single_sku_large` | single-SKU | 1 | 50 | BOX_S, BOX_M, BOX_L |
| 6 | `single_sku_100` | single-SKU | 1 | 100 | BOX_S, BOX_M, BOX_L |
| 7 | `multi_sku_100` | multi-SKU | 10 | ~100 | BOX_S, BOX_M, BOX_L |

Box types used across all scenarios:

| ID | Name | Inner dims (L×W×H cm) | Cost |
|----|------|----------------------|------|
| BOX_S | Pacco-S | 15×12×10 | 1.0 |
| BOX_M | Pacco-M | 25×20×15 | 2.0 |
| BOX_L | Pacco-L | 35×28×20 | 3.5 |

All scenarios use `grid_resolution=1.0`.

---

## Console Output

```
=== 3D Bin Packing Benchmark — 2026-04-10 15:30:00 ===

Scenario             | Mode       | Reps | Min(s)  | Max(s)  | Avg(s)  | Boxes | BOX_S | BOX_M | BOX_L | Unassigned
---------------------|------------|------|---------|---------|---------|-------|-------|-------|-------|----------
single_sku_tiny      | single-SKU |    5 |   0.002 |   0.004 |   0.003 |     1 |     1 |     0 |     0 |         0
single_sku_medium    | single-SKU |    5 |   0.015 |   0.020 |   0.017 |     2 |     0 |     2 |     0 |         0
...

Results saved to: dbin_local/benchmark_results/
```

---

## CSV Schema

File: `benchmark_results.csv`

Columns:
```
timestamp, scenario, mode, repetitions, min_s, max_s, avg_s,
total_boxes, box_breakdown, unassigned_items
```

- `timestamp`: ISO 8601 (e.g. `2026-04-10T15:30:00`)
- `box_breakdown`: compact string, e.g. `BOX_S:1,BOX_M:2,BOX_L:0`
- Each run appends rows (one per scenario); existing file is not overwritten

---

## JSON Schema

File: `benchmark_results.json`

Structure: a JSON array; each run appends one object:

```json
{
  "run_timestamp": "2026-04-10T15:30:00",
  "scenarios": [
    {
      "name": "single_sku_tiny",
      "mode": "single-SKU",
      "repetitions": 5,
      "times_s": [0.002, 0.003, 0.003, 0.004, 0.003],
      "min_s": 0.002,
      "max_s": 0.004,
      "avg_s": 0.003,
      "total_boxes": 1,
      "box_breakdown": {"BOX_S": 1, "BOX_M": 0, "BOX_L": 0},
      "unassigned_items": 0
    }
  ]
}
```

---

## Dependencies

No new packages required. Uses only:
- `time` (perf_counter)
- `json`
- `csv`
- `pathlib`
- `datetime`
- `collections.Counter`

`BinCore` is imported via the same relative path used in `demo_ultimate.py`.
