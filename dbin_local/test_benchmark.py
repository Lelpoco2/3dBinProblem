# dbin_local/test_benchmark.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import csv as csv_module
import json
from unittest.mock import patch

import pytest

from benchmark import (
    _run_scenario_once,
    _run_benchmark,
    _write_csv,
    _write_json,
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


def test_write_json_corrupted_file_recovers(tmp_path):
    json_path = tmp_path / "benchmark_results.json"
    json_path.write_text("{ invalid json }", encoding="utf-8")
    with patch("benchmark.RESULTS_DIR", tmp_path):
        _write_json(_sample_results(), "2026-04-10T10:00:00")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["scenarios"][0]["name"] == "test_scenario"
