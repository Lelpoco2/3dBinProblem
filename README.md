
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

## TODOs
- Improving the packing heuristics (e.g., better item ordering, more complex placement strategies).
- Adding FastAPI endpoints for web-based interaction.

## dbin_api (FastAPI wrapper)

The `dbin_api` package is a refactored FastAPI-based wrapper around the original packing engine. It exposes the packing functionality over HTTP and provides modular code for the core algorithms.

Key files in `dbin_api/`:
- `dbin_api/models.py` — dataclasses for the core packing engine and Pydantic request/response models.
- `dbin_api/packing.py` — core packing algorithm (geometry helpers, multi-SKU packer, single-SKU planner).
- `dbin_api/api.py` — FastAPI application exposing HTTP endpoints.
- `dbin_api/test_packing.py` — example pytest tests including API endpoint checks.

Available endpoints
- `GET /` — Service info (name and version).
- `GET /health` — Health check.
- `POST /pack` — Multi-SKU packer.
  - Request JSON: `{ "items": [...], "box_types": [...], "grid_resolution": 1.0 }`
  - Returns: `PackingResult` (boxes with placed items, unassigned items, summary).
- `POST /pack/single-sku` — Single-SKU planner + packer.
  - Request JSON: `{ "item": {...}, "box_types": [...], "grid_resolution": 1.0 }`
  - Returns: packing result plus planned box mix and summary.

Schemas
- `ItemCreate` — item input (id, length, width, height, quantity, can_rotate, name).
- `BoxTypeCreate` — box input (id, inner_length, inner_width, inner_height, cost, max_boxes, effective_volume, container_type, name).
- `PackingResult` — response that includes `boxes` (with placed items) and `unassigned_items`.

Running the API locally
1. Install dependencies (recommended):
```bash
python -m pip install fastapi uvicorn pydantic pytest
# plus optional plotting/test deps:
python -m pip install numpy matplotlib
```

2. Start the server from the project root:
```bash
uvicorn dbin_api.api:app --reload --host 127.0.0.1 --port 8000
```

3. Open the interactive API docs:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

Examples

Multi-SKU pack (POST /pack) — request body:
```json
{
  "items": [
    { "id": "SKU-1", "length": 4.0, "width": 3.0, "height": 2.0, "quantity": 2, "can_rotate": true, "name": "A" },
    { "id": "SKU-2", "length": 6.0, "width": 5.0, "height": 2.0, "quantity": 1, "can_rotate": true, "name": "B" }
  ],
  "box_types": [
    { "id": "BOX-1", "inner_length": 20.0, "inner_width": 15.0, "inner_height": 10.0, "cost": 1.0, "max_boxes": null, "effective_volume": null, "container_type": "BOX", "name": "StandardBox" }
  ],
  "grid_resolution": 1.0
}
```

Single-SKU pack (POST /pack/single-sku) — request body:
```json
{
  "item": { "id": "SKU-X", "length": 4.0, "width": 3.0, "height": 2.0, "quantity": 6, "can_rotate": true, "name": "X" },
  "box_types": [
    { "id": "BOX-1", "inner_length": 20.0, "inner_width": 15.0, "inner_height": 10.0, "cost": 1.0, "max_boxes": null, "effective_volume": null, "container_type": "BOX", "name": "StandardBox" }
  ],
  "grid_resolution": 1.0
}
```

Curl example
```bash
curl -sS -X POST http://127.0.0.1:8000/pack \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON'
{
  "items": [{ "id": "SKU-1", "length": 4, "width": 3, "height": 2, "quantity": 2 }],
  "box_types": [{ "id": "BOX-1", "inner_length": 20, "inner_width": 15, "inner_height": 10, "cost": 1.0 }],
  "grid_resolution": 1.0
}
JSON
```

Testing
- Unit tests and API tests live in `dbin_api/test_packing.py`.
- Run tests:
```bash
pytest -q
```

Notes
- The FastAPI wrapper exposes the geometry-only packer for experimentation and integration. For production use you may want to add authentication, request rate limits, background job handling for long runs, and stronger input validation depending on your use case.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

Copyright (c) 2026 Lelpoco2


