
# 3D Bin Packing (3dBinProblem)

This repository provides a small Python toolkit and demos for exploring a 3D bin packing / item placement problem with simple visualization. It contains core packing logic, example item sets, SKU-based demos, and a lightweight 3D plotter for visual verification.

## Features

- Core packing utilities implemented in `BinCore.py`.
- Example scripts demonstrating different input styles: `demo_item_list.py`, `demo_sku.py`, and `demo_ultimate.py`.
- A simple 3D plotter for visualizing placements in `plotter3d.py`.

## Requirements

- Python 3.8+
- Recommended (install via pip):

```bash
python -m pip install numpy matplotlib
```

The project aims to be lightweight; if you prefer other plotting or numeric libraries you can adapt `plotter3d.py` accordingly.

## Files Overview

- `BinCore.py` — Core algorithms and data structures for bin/item handling and placement heuristics.
- `demo_item_list.py` — Demo that uses a simple list of item dimensions to pack into bins.
- `demo_sku.py` — Demo showing SKU-style inputs (quantities and sizes) and packing them.
- `demo_ultimate.py` — A combined / advanced demo that exercises more features and larger item sets.
- `plotter3d.py` — Minimal 3D visualization helper using `matplotlib`'s 3D axes to draw bins and items.

## Usage

Run any demo directly from the command line. Each demo prints placement results to the console and (where applicable) opens a visualization window:

```bash
python demo_item_list.py
python demo_sku.py
python demo_ultimate.py
python plotter3d.py   # standalone visual tests
```

Typical workflow:

1. Prepare items (dimensions and optionally quantities).
2. Use `BinCore` functions to pack items into bins.
3. Inspect textual results or call `plotter3d` helpers to render placements.

## Example (pseudo)

```py
from BinCore import pack_items
from plotter3d import draw_packing

bins = pack_items(items, bin_size=(100,100,100))
draw_packing(bins)
```

Adjust the call signatures to match the functions in `BinCore.py` (the demos show concrete usage).

## Extending & Development

- Algorithms: `BinCore.py` is designed to be readable and modified — try different heuristics or ordering strategies.
- Visualization: `plotter3d.py` is intentionally minimal; swap in `pythreejs`, `vtk`, or other libraries for more advanced rendering.
- Tests: Add unit tests for packing invariants (no overlaps, fit within bin bounds, correct counts).

## Notes & Limitations

- This repo provides an educational and experimental implementation rather than a production-grade solver.
- Performance for large inputs is not optimized; consider more advanced bin-packing libraries or integer programming for heavy workloads.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

Copyright (c) 2026 Lelpoco2


