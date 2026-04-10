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
