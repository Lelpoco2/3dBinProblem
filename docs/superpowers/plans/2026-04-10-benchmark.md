# Benchmark System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `dbin_local/benchmark.py` — uno script standalone che misura la velocità dell'algoritmo di packing su 7 scenari di complessità crescente, stampando i risultati a console e salvandoli come CSV + JSON.

**Architecture:** File unico con tre strati: (1) definizione scenari (items + box types), (2) runner che cronometra N ripetizioni e calcola statistiche, (3) layer di output che formatta la tabella a console, scrive CSV e JSON. Usa la logica `smart_pack` (replicata localmente da `demo_ultimate.py`) per scegliere automaticamente single-SKU vs multi-SKU.

**Tech Stack:** Solo stdlib Python (`time`, `json`, `csv`, `pathlib`, `datetime`, `collections.Counter`). `BinCore` per la logica di packing.

---

## File Structure

| File | Azione | Responsabilità |
|------|--------|----------------|
| `dbin_local/benchmark.py` | Create | Script principale: scenari, runner, output |
| `dbin_local/test_benchmark.py` | Create | Unit test per runner e output helpers |
| `dbin_local/benchmark_results/` | Auto-created | Directory output (creata dallo script) |

---

## Task 1: Test + implementa `_run_scenario_once`

**Files:**
- Create: `dbin_local/benchmark.py` (scheletro + funzione)
- Create: `dbin_local/test_benchmark.py`

- [ ] **Step 1: Crea lo scheletro di `benchmark.py` con imports e box types condivisi**

```python
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
```

- [ ] **Step 2: Scrivi il test per `_run_scenario_once` in `test_benchmark.py`**

```python
# dbin_local/test_benchmark.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from benchmark import (
    _run_scenario_once,
    _BOX_S, _BOX_M, _BOXES_SM,
)
from BinCore import ItemType


def _tiny_scenario():
    return {
        "name": "test_tiny",
        "items": [ItemType("SKU_T", 5, 4, 3, quantity=2, name="Test")],
        "box_types": _BOXES_SM,
    }


def test_run_scenario_once_shape():
    result = _run_scenario_once(_tiny_scenario())
    assert "mode" in result
    assert "total_boxes" in result
    assert "box_breakdown" in result
    assert "unassigned_items" in result


def test_run_scenario_once_single_sku_mode():
    result = _run_scenario_once(_tiny_scenario())
    assert result["mode"] == "single-SKU"


def test_run_scenario_once_multi_sku_mode():
    scenario = {
        "name": "test_multi",
        "items": [
            ItemType("SKU_A", 5, 4, 3, quantity=1, name="A"),
            ItemType("SKU_B", 8, 6, 4, quantity=1, name="B"),
        ],
        "box_types": _BOXES_SM,
    }
    result = _run_scenario_once(scenario)
    assert result["mode"] == "multi-SKU"


def test_run_scenario_once_breakdown_keys():
    result = _run_scenario_once(_tiny_scenario())
    assert "BOX_S" in result["box_breakdown"]
    assert "BOX_M" in result["box_breakdown"]


def test_run_scenario_once_no_unassigned():
    result = _run_scenario_once(_tiny_scenario())
    assert result["unassigned_items"] == 0
```

- [ ] **Step 3: Esegui il test — attendi FAIL ("cannot import `_run_scenario_once`")**

```
cd dbin_local
pytest test_benchmark.py::test_run_scenario_once_shape -v
```

Atteso: `ImportError` o `ModuleNotFoundError`.

- [ ] **Step 4: Implementa `_smart_pack` e `_run_scenario_once` in `benchmark.py`**

Aggiungi dopo le costanti di box:

```python
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
```

- [ ] **Step 5: Esegui i test — attendi tutti PASS**

```
cd dbin_local
pytest test_benchmark.py -v
```

Atteso: 5 test PASS.

- [ ] **Step 6: Commit**

```bash
git add dbin_local/benchmark.py dbin_local/test_benchmark.py
git commit -m "feat: add benchmark skeleton and _run_scenario_once"
```

---

## Task 2: Test + implementa `_run_benchmark`

**Files:**
- Modify: `dbin_local/benchmark.py`
- Modify: `dbin_local/test_benchmark.py`

- [ ] **Step 1: Scrivi il test per `_run_benchmark`**

Aggiungi in fondo a `test_benchmark.py`:

```python
from benchmark import _run_benchmark


def test_run_benchmark_shape():
    result = _run_benchmark(_tiny_scenario(), repetitions=2)
    for key in ("name", "mode", "repetitions", "times_s", "min_s", "max_s", "avg_s",
                "total_boxes", "box_breakdown", "unassigned_items"):
        assert key in result, f"Missing key: {key}"


def test_run_benchmark_repetitions_count():
    result = _run_benchmark(_tiny_scenario(), repetitions=3)
    assert result["repetitions"] == 3
    assert len(result["times_s"]) == 3


def test_run_benchmark_stats_consistent():
    result = _run_benchmark(_tiny_scenario(), repetitions=4)
    assert result["min_s"] <= result["avg_s"] <= result["max_s"]
    assert result["min_s"] == min(result["times_s"])
    assert result["max_s"] == max(result["times_s"])
    assert abs(result["avg_s"] - sum(result["times_s"]) / 4) < 1e-9


def test_run_benchmark_times_positive():
    result = _run_benchmark(_tiny_scenario(), repetitions=2)
    assert all(t > 0 for t in result["times_s"])
```

- [ ] **Step 2: Esegui i test — attendi FAIL ("cannot import `_run_benchmark`")**

```
cd dbin_local
pytest test_benchmark.py::test_run_benchmark_shape -v
```

Atteso: `ImportError`.

- [ ] **Step 3: Implementa `_run_benchmark` in `benchmark.py`**

Aggiungi dopo `_run_scenario_once`:

```python
def _run_benchmark(scenario: dict, repetitions: int = REPETITIONS) -> dict:
    times = []
    result = None
    for _ in range(repetitions):
        t0 = time.perf_counter()
        result = _run_scenario_once(scenario)
        t1 = time.perf_counter()
        times.append(round(t1 - t0, 6))
    return {
        "name": scenario["name"],
        "mode": result["mode"],
        "repetitions": repetitions,
        "times_s": times,
        "min_s": round(min(times), 6),
        "max_s": round(max(times), 6),
        "avg_s": round(sum(times) / len(times), 6),
        "total_boxes": result["total_boxes"],
        "box_breakdown": result["box_breakdown"],
        "unassigned_items": result["unassigned_items"],
    }
```

- [ ] **Step 4: Esegui tutti i test — attendi PASS**

```
cd dbin_local
pytest test_benchmark.py -v
```

Atteso: 9 test PASS.

- [ ] **Step 5: Commit**

```bash
git add dbin_local/benchmark.py dbin_local/test_benchmark.py
git commit -m "feat: add _run_benchmark with timing and stats"
```

---

## Task 3: Test + implementa output CSV e JSON

**Files:**
- Modify: `dbin_local/benchmark.py`
- Modify: `dbin_local/test_benchmark.py`

- [ ] **Step 1: Scrivi test per `_write_csv` e `_write_json`**

Aggiungi in fondo a `test_benchmark.py`:

```python
import csv as csv_module
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from benchmark import _write_csv, _write_json


def _sample_results():
    return [
        {
            "name": "test_scenario",
            "mode": "single-SKU",
            "repetitions": 2,
            "times_s": [0.01, 0.02],
            "min_s": 0.01,
            "max_s": 0.02,
            "avg_s": 0.015,
            "total_boxes": 1,
            "box_breakdown": {"BOX_S": 1, "BOX_M": 0},
            "unassigned_items": 0,
        }
    ]


def test_write_csv_creates_file(tmp_path):
    with patch("benchmark.RESULTS_DIR", tmp_path):
        _write_csv(_sample_results(), "2026-04-10T10:00:00")
    csv_path = tmp_path / "benchmark_results.csv"
    assert csv_path.exists()


def test_write_csv_header_and_row(tmp_path):
    with patch("benchmark.RESULTS_DIR", tmp_path):
        _write_csv(_sample_results(), "2026-04-10T10:00:00")
    csv_path = tmp_path / "benchmark_results.csv"
    with csv_path.open() as f:
        reader = csv_module.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["scenario"] == "test_scenario"
    assert rows[0]["mode"] == "single-SKU"
    assert "BOX_S:1" in rows[0]["box_breakdown"]


def test_write_csv_appends(tmp_path):
    with patch("benchmark.RESULTS_DIR", tmp_path):
        _write_csv(_sample_results(), "2026-04-10T10:00:00")
        _write_csv(_sample_results(), "2026-04-10T10:01:00")
    csv_path = tmp_path / "benchmark_results.csv"
    with csv_path.open() as f:
        reader = csv_module.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2


def test_write_json_creates_file(tmp_path):
    with patch("benchmark.RESULTS_DIR", tmp_path):
        _write_json(_sample_results(), "2026-04-10T10:00:00")
    json_path = tmp_path / "benchmark_results.json"
    assert json_path.exists()


def test_write_json_structure(tmp_path):
    with patch("benchmark.RESULTS_DIR", tmp_path):
        _write_json(_sample_results(), "2026-04-10T10:00:00")
    json_path = tmp_path / "benchmark_results.json"
    with json_path.open() as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 1
    run = data[0]
    assert "run_timestamp" in run
    assert "scenarios" in run
    assert run["scenarios"][0]["name"] == "test_scenario"
    assert isinstance(run["scenarios"][0]["box_breakdown"], dict)


def test_write_json_appends_runs(tmp_path):
    with patch("benchmark.RESULTS_DIR", tmp_path):
        _write_json(_sample_results(), "2026-04-10T10:00:00")
        _write_json(_sample_results(), "2026-04-10T10:01:00")
    json_path = tmp_path / "benchmark_results.json"
    with json_path.open() as f:
        data = json.load(f)
    assert len(data) == 2
```

- [ ] **Step 2: Esegui i test — attendi FAIL**

```
cd dbin_local
pytest test_benchmark.py::test_write_csv_creates_file -v
```

Atteso: `ImportError`.

- [ ] **Step 3: Implementa `_write_csv` in `benchmark.py`**

```python
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
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in results:
            breakdown_str = ",".join(f"{k}:{v}" for k, v in r["box_breakdown"].items())
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
```

- [ ] **Step 4: Implementa `_write_json` in `benchmark.py`**

```python
# ---------------------------------------------------------------------------
# Output — JSON
# ---------------------------------------------------------------------------

def _write_json(results: list, timestamp: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "benchmark_results.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            all_runs = json.load(f)
    else:
        all_runs = []
    all_runs.append({"run_timestamp": timestamp, "scenarios": results})
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)
```

- [ ] **Step 5: Esegui tutti i test — attendi PASS**

```
cd dbin_local
pytest test_benchmark.py -v
```

Atteso: 17 test PASS.

- [ ] **Step 6: Commit**

```bash
git add dbin_local/benchmark.py dbin_local/test_benchmark.py
git commit -m "feat: add CSV and JSON output writers with tests"
```

---

## Task 4: Console output, 7 scenari e `main()`

**Files:**
- Modify: `dbin_local/benchmark.py`

- [ ] **Step 1: Aggiungi `_print_results` in `benchmark.py`**

```python
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
```

- [ ] **Step 2: Aggiungi i 7 scenari e `main()` in `benchmark.py`**

```python
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
```

- [ ] **Step 3: Esegui tutti i test per regressioni**

```
cd dbin_local
pytest test_benchmark.py -v
```

Atteso: 17 test PASS, nessuna regressione.

- [ ] **Step 4: Esegui lo script per smoke test manuale**

```
cd dbin_local
python benchmark.py
```

Atteso: tabella stampata a console + file creati in `benchmark_results/`.

Verifica:
```
dir benchmark_results
```
Devono esistere `benchmark_results.csv` e `benchmark_results.json`.

- [ ] **Step 5: Commit finale**

```bash
git add dbin_local/benchmark.py dbin_local/test_benchmark.py
git commit -m "feat: add 7-scenario benchmark with console, CSV and JSON output"
```

---

## Self-Review

**Spec coverage:**
- ✅ 7 scenari di complessità crescente (1→7)
- ✅ Logica smart_pack (single-SKU / multi-SKU routing)
- ✅ 5 ripetizioni per scenario
- ✅ Output console con tabella min/max/avg + dettaglio per box type
- ✅ CSV append con tutti i campi richiesti
- ✅ JSON append con struttura per run
- ✅ `box_breakdown` per tipo box in tutti gli output
- ✅ Nessuna dipendenza esterna

**Type consistency:** `_run_scenario_once` → `_run_benchmark` → `_print_results` / `_write_csv` / `_write_json` usano tutti la stessa shape dict. Chiavi coerenti tra tutti i task.

**Placeholder scan:** Nessun TBD o TODO.
