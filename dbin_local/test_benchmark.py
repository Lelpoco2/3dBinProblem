# dbin_local/test_benchmark.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pytest

from benchmark import (
    _run_scenario_once,
    _run_benchmark,
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


def test_run_scenario_once_with_unassigned():
    scenario = {
        "name": "test_unassigned",
        "items": [ItemType("SKU_HUGE", 999, 999, 999, quantity=1, name="Huge")],
        "box_types": _BOXES_SM,
    }
    result = _run_scenario_once(scenario)
    assert result["unassigned_items"] == 1
    assert result["total_boxes"] == 0


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


def test_run_benchmark_zero_repetitions_raises():
    with pytest.raises(ValueError, match="repetitions"):
        _run_benchmark(_tiny_scenario(), repetitions=0)
