"""
Tests for the dbin_api packing core and FastAPI endpoints.

These are example pytest-style tests that:
- Validate the low-level packing functions behave reasonably on small inputs.
- Validate the FastAPI endpoints exposed by `dbin_api.api` using TestClient.

Run with: pytest -q

Notes:
- Tests will skip API client tests if FastAPI/TestClient dependencies are not available.
- The tests keep dimensions and quantities small so they run quickly.
"""

import pytest

# Import the core modules from the refactor
from dbin_api import models as m
from dbin_api import packing as packing_core

# For API tests, ensure fastapi + testclient available; otherwise skip those tests.
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # type: ignore

from dbin_api import api as api_module


def example_box_types():
    """
    Helper to create a couple of example BoxType instances (dataclasses).
    Returns a list of dataclass BoxType objects and corresponding dicts for API calls.
    """
    bt1 = m.BoxType(
        id="BOX_SMALL",
        inner_length=10.0,
        inner_width=8.0,
        inner_height=4.0,
        cost=0.5,
        max_boxes=None,
        effective_volume=None,
        container_type="BOX",
        name="SmallBox",
    )
    bt2 = m.BoxType(
        id="BOX_MED",
        inner_length=20.0,
        inner_width=15.0,
        inner_height=10.0,
        cost=1.0,
        max_boxes=None,
        effective_volume=None,
        container_type="BOX",
        name="MediumBox",
    )
    # Dict representations (suitable for sending as JSON to the API)
    bt1_dict = {
        "id": bt1.id,
        "inner_length": bt1.inner_length,
        "inner_width": bt1.inner_width,
        "inner_height": bt1.inner_height,
        "cost": bt1.cost,
        "max_boxes": bt1.max_boxes,
        "effective_volume": bt1.effective_volume,
        "container_type": bt1.container_type,
        "name": bt1.name,
    }
    bt2_dict = {
        "id": bt2.id,
        "inner_length": bt2.inner_length,
        "inner_width": bt2.inner_width,
        "inner_height": bt2.inner_height,
        "cost": bt2.cost,
        "max_boxes": bt2.max_boxes,
        "effective_volume": bt2.effective_volume,
        "container_type": bt2.container_type,
        "name": bt2.name,
    }
    return [bt1, bt2], [bt1_dict, bt2_dict]


def example_items():
    """
    Helper to create example ItemType dataclasses and dicts for API.
    """
    it1 = m.ItemType(
        id="SKU_A",
        length=4.0,
        width=3.0,
        height=2.0,
        quantity=2,
        can_rotate=True,
        name="A",
    )
    it2 = m.ItemType(
        id="SKU_B",
        length=6.0,
        width=5.0,
        height=2.0,
        quantity=1,
        can_rotate=True,
        name="B",
    )
    it1_dict = {
        "id": it1.id,
        "length": it1.length,
        "width": it1.width,
        "height": it1.height,
        "quantity": it1.quantity,
        "can_rotate": it1.can_rotate,
        "name": it1.name,
    }
    it2_dict = {
        "id": it2.id,
        "length": it2.length,
        "width": it2.width,
        "height": it2.height,
        "quantity": it2.quantity,
        "can_rotate": it2.can_rotate,
        "name": it2.name,
    }
    return [it1, it2], [it1_dict, it2_dict]


def test_pack_order_basic():
    """
    Basic test of the multi-SKU packer core function.
    We expect that small items fit into the medium box and no items are left unassigned.
    """
    items_dc, _ = example_items()
    box_types_dc, _ = example_box_types()

    boxes, unassigned = packing_core.pack_order(
        items_dc, box_types_dc, grid_resolution=1.0
    )

    # Basic assertions: we should get at least one box and no unassigned items
    assert isinstance(boxes, list)
    assert isinstance(unassigned, list)
    assert len(unassigned) == 0, (
        f"Unexpected unassigned items: {[u.id for u in unassigned]}"
    )

    # Check each placed item is a PlacedItem and references a known box id
    for b in boxes:
        assert isinstance(b.items, list)
        for placed in b.items:
            assert placed.box_id.startswith(b.box_type.id)
            # rotation should be a 3-tuple
            assert len(placed.rotation) == 3


def test_pack_single_sku_planner():
    """
    Test the single-SKU planner + packer for a number of identical small items.
    We verify that a mix is proposed and that the packer returns boxes covering many items.
    """
    # Define a single SKU with quantity 6
    item = m.ItemType(
        id="SKU_X",
        length=4.0,
        width=3.0,
        height=2.0,
        quantity=6,
        can_rotate=True,
        name="X",
    )
    # Reuse example box types
    box_types_dc, _ = example_box_types()

    boxes, unassigned, planned_box_types, mix = packing_core.pack_single_sku_order(
        item, box_types_dc, grid_resolution=1.0
    )

    # mix should be a dict mapping box ids to planned counts
    assert isinstance(mix, dict)
    # planned_box_types should have same length as provided box types
    assert len(planned_box_types) == len(box_types_dc)

    # All outputs should be lists/dicts of reasonable types
    assert isinstance(boxes, list)
    assert isinstance(unassigned, list)

    # The total number of placed items plus unassigned should not exceed requested quantity
    placed_count = sum(len(b.items) for b in boxes)
    assert placed_count + len(unassigned) <= item.quantity


@pytest.fixture
def client():
    """
    FastAPI TestClient fixture for API tests.
    """
    return TestClient(api_module.app)


def test_api_pack_multi(client):
    """
    Test the POST /pack endpoint using the TestClient.
    Validates response shape and some content.
    """
    _, items_dicts = example_items()
    _, boxes_dicts = example_box_types()

    payload = {
        "items": items_dicts,
        "box_types": boxes_dicts,
        "grid_resolution": 1.0,
    }

    response = client.post("/pack", json=payload)
    assert response.status_code == 200, (
        f"API error: {response.status_code} - {response.text}"
    )

    data = response.json()
    # Verify expected top-level keys present
    assert "boxes" in data
    assert "unassigned_items" in data
    assert isinstance(data["boxes"], list)
    assert isinstance(data["unassigned_items"], list)

    # If boxes returned, verify expected fields inside first box
    if data["boxes"]:
        first_box = data["boxes"][0]
        assert "items" in first_box
        assert "used_volume" in first_box
        assert "capacity_volume" in first_box
        assert "utilization" in first_box


def test_api_pack_single_sku(client):
    """
    Test the POST /pack/single-sku endpoint.
    """
    # Single item request (quantity > 1)
    item = {
        "id": "SKU_TEST",
        "length": 4.0,
        "width": 3.0,
        "height": 2.0,
        "quantity": 4,
        "can_rotate": True,
        "name": "TestSKU",
    }
    _, boxes_dicts = example_box_types()

    payload = {
        "item": item,
        "box_types": boxes_dicts,
        "grid_resolution": 1.0,
    }

    response = client.post("/pack/single-sku", json=payload)
    assert response.status_code == 200, (
        f"API single-sku error: {response.status_code} - {response.text}"
    )

    data = response.json()
    # The response encloses 'packing_result' and 'mix' keys per implementation
    assert "packing_result" in data
    assert "mix" in data

    # packing_result should itself contain boxes/unassigned_items (serialized)
    packing_result = data["packing_result"]
    assert hasattr(packing_result, "dict") or isinstance(packing_result, dict)
    # Access as dict if necessary
    if isinstance(packing_result, dict):
        assert "boxes" in packing_result
        assert "unassigned_items" in packing_result
