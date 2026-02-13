"""
FastAPI application exposing the 3D bin-packing functionality.

This module provides a small, well-documented API surface built on top of the
refactored packing engine in dbin_api.

Endpoints:
- GET /health
- GET / (service info / version)
- POST /pack           -> multi-SKU packer (general list of items)
- POST /pack/single-sku -> single-SKU planner + packer

Notes:
- The API uses the Pydantic request/response models defined in `dbin_api.models`.
- The computational core remains pure-Python and uses dataclasses (in `dbin_api.models`)
  and the algorithm implementation in `dbin_api.packing`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import __version__ as PACKAGE_VERSION
from . import packing as packing_core
from .models import (
    BoxTypeCreate,
    ItemCreate,
    PackingResult,
    boxtypecreate_to_dataclass,
    itemcreate_to_dataclass,
    packing_result_from_dataclasses,
)

logger = logging.getLogger("dbin_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="dbin_api - 3D bin packing",
    version=PACKAGE_VERSION,
    description="API wrapper around the 3D geometry-only bin packing engine.",
)

# Allow cross-origin calls for common dev scenarios (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Health & info endpoints
# ---------------------------


@app.get("/", summary="Service info")
async def root() -> Dict[str, Any]:
    """
    Basic service information and version.
    """
    return {"service": "dbin_api", "version": PACKAGE_VERSION}


@app.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}


# ---------------------------
# Request bodies
# ---------------------------


class PackRequest:
    """
    Lightweight request wrapper type used inside handlers.
    We don't use this as a Pydantic model to keep the handler signatures
    explicit (FastAPI will still accept typed bodies when using function
    parameters directly). This class documents expected fields:
    - items: list of ItemCreate
    - box_types: list of BoxTypeCreate
    - grid_resolution: optional float controlling placement grid step
    """

    items: List[ItemCreate]
    box_types: List[BoxTypeCreate]
    grid_resolution: Optional[float] = 1.0


# ---------------------------
# Packing endpoints
# ---------------------------


@app.post(
    "/pack",
    response_model=PackingResult,
    summary="Pack a list of items into available box types (multi-SKU)",
)
async def pack_items(
    items: List[ItemCreate],
    box_types: List[BoxTypeCreate],
    grid_resolution: Optional[float] = 1.0,
) -> PackingResult:
    """
    Multi-SKU packing endpoint.

    Request:
    - items: list of SKUs (ItemCreate). Each item may have `quantity` > 1.
    - box_types: list of available boxes (BoxTypeCreate).
    - grid_resolution: optional placement grid resolution (default 1.0).

    Response:
    - PackingResult: boxes (with placed items), unassigned_items, and an optional summary.
    """
    if not items:
        raise HTTPException(status_code=400, detail="`items` must be a non-empty list.")
    if not box_types:
        raise HTTPException(
            status_code=400, detail="`box_types` must be a non-empty list."
        )

    # Convert Pydantic -> dataclasses used by the core
    dataclass_items = [itemcreate_to_dataclass(it) for it in items]
    dataclass_box_types = [boxtypecreate_to_dataclass(bt) for bt in box_types]

    logger.info(
        "pack_items called: %d items, %d box_types, grid_resolution=%s",
        len(dataclass_items),
        len(dataclass_box_types),
        grid_resolution,
    )

    # Run the core packer (CPU-bound). Keep synchronous call - FastAPI will handle threadpool
    boxes, unassigned = packing_core.pack_order(
        dataclass_items, dataclass_box_types, grid_resolution=grid_resolution or 1.0
    )

    summary: Dict[str, Any] = {
        "requested_items": sum(it.quantity for it in dataclass_items),
        "box_types_provided": [bt.id for bt in dataclass_box_types],
        "boxes_opened": len(boxes),
        "unassigned_count": len(unassigned),
    }

    result = packing_result_from_dataclasses(
        boxes=boxes, unassigned=unassigned, summary=summary
    )
    return result


@app.post(
    "/pack/single-sku",
    summary="Plan and pack a single-SKU order (returns planned mix + placement)",
)
async def pack_single_sku(
    item: ItemCreate,
    box_types: List[BoxTypeCreate],
    grid_resolution: Optional[float] = 1.0,
) -> Dict[str, Any]:
    """
    Single-SKU planner + packer.

    Request:
    - item: ItemCreate (with `quantity` set to desired total)
    - box_types: list of available BoxTypeCreate
    - grid_resolution: optional resolution for placement phase

    Response (dict):
    - boxes: list of placed boxes (as in PackingResult)
    - unassigned_items: list of unassigned items
    - planned_box_types: list of box types used for the packing run (with max_boxes set per plan)
    - mix: dict of box_type_id -> planned count
    - summary: metadata
    """
    if item.quantity <= 0:
        raise HTTPException(status_code=400, detail="`item.quantity` must be > 0")

    if not box_types:
        raise HTTPException(status_code=400, detail="`box_types` must be provided")

    dataclass_item = itemcreate_to_dataclass(item)
    dataclass_box_types = [boxtypecreate_to_dataclass(bt) for bt in box_types]

    logger.info(
        "pack_single_sku called: item=%s qty=%d, box_types=%d, grid_resolution=%s",
        dataclass_item.id,
        dataclass_item.quantity,
        len(dataclass_box_types),
        grid_resolution,
    )

    boxes, unassigned, planned_box_types, mix = packing_core.pack_single_sku_order(
        dataclass_item, dataclass_box_types, grid_resolution=grid_resolution or 1.0
    )

    # Convert dataclass outputs to pydantic PackingResult for the `boxes` + `unassigned` section
    packing_result = packing_result_from_dataclasses(
        boxes=boxes, unassigned=unassigned, summary=None
    )

    # Build response payload with additional planner data
    response: Dict[str, Any] = {
        "packing_result": packing_result,
        "planned_box_types": [
            {
                "id": bt.id,
                "inner_length": bt.inner_length,
                "inner_width": bt.inner_width,
                "inner_height": bt.inner_height,
                "cost": bt.cost,
                "max_boxes": bt.max_boxes,
                "effective_volume": bt.effective_volume,
                "container_type": bt.container_type,
                "name": bt.name,
            }
            for bt in planned_box_types
        ],
        "mix": mix,
        "summary": {
            "requested_quantity": dataclass_item.quantity,
            "boxes_opened": len(boxes),
            "unassigned_count": len(unassigned),
        },
    }

    return response


# ---------------------------
# Exception handlers & utilities
# ---------------------------


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    # Basic generic handler to ensure JSON responses for unexpected errors.
    logger.exception("Unhandled exception: %s", exc)
    return HTTPException(status_code=500, detail="Internal server error")
