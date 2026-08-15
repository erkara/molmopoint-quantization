import pytest

from evaluate import (
    OFFICIAL_MOLMO2_COMMIT,
    default_run_name,
    parse_variants,
    select_positions,
    summarize_records,
)


def _record(source_index, *, category=None, metrics, latency=1.0, memory=8.0):
    return {
        "source_index": source_index,
        "status": "passed",
        "category": category,
        "metrics": metrics,
        "inference_seconds": latency,
        "peak_vram_gib": memory,
    }


def test_five_percent_selection_is_deterministic_and_shared():
    first = select_positions(982, fraction=0.05, seed=0)
    second = select_positions(982, fraction=0.05, seed=0)
    assert first == second
    assert len(first) == 49
    assert first == sorted(first)


def test_selection_and_variant_inputs_are_validated():
    assert parse_variants("all") == ["bf16", "int4", "int8"]
    assert parse_variants("int4,bf16,int4") == ["int4", "bf16"]
    with pytest.raises(ValueError, match="fraction"):
        select_positions(10, fraction=0)
    with pytest.raises(ValueError, match="Invalid variants"):
        parse_variants("int3")


def test_default_run_name_records_scope_and_seed():
    assert (
        default_run_name("pointbench", fraction=0.05, max_examples=None, seed=7)
        == "pointbench-5pct-seed7"
    )


def test_official_summary_uses_equal_category_weighting():
    records = [
        _record(0, category="affordable", metrics={"accuracy": 1.0}),
        _record(1, category="affordable", metrics={"accuracy": 0.0}),
        _record(2, category="counting", metrics={"accuracy": 1.0}),
    ]
    summary = summarize_records(
        records,
        task="point_bench",
        variant="int4",
        model="final-int4",
        official_commit=OFFICIAL_MOLMO2_COMMIT,
        expected_examples=3,
        public_source_examples=3,
        selected_examples=3,
        max_new_tokens=256,
    )
    assert summary["metrics"] == {
        "categories": {"affordable": 0.5, "counting": 1.0},
        "average": 0.75,
    }
    assert summary["evaluation_complete"] is True
    assert summary["protocol_coverage"] == "full"


def test_pointing_summary_marks_partial_public_recovery():
    records = [
        _record(7, metrics={"precision": 1.0, "recall": 0.5, "f1": 2 / 3}, memory=9.0),
        _record(9, metrics={"precision": 0.5, "recall": 1.0, "f1": 2 / 3}),
    ]
    summary = summarize_records(
        records,
        task="pointing_eval_v2",
        variant="int8",
        model="final-int8",
        official_commit=OFFICIAL_MOLMO2_COMMIT,
        expected_examples=4,
        public_source_examples=2,
        selected_examples=2,
        max_new_tokens=192,
    )
    assert summary["metrics"] == {"precision": 0.75, "recall": 0.75, "f1": 0.666667}
    assert summary["coverage_fraction"] == 0.5
    assert summary["protocol_coverage"] == "partial-public-recovery"
    assert summary["performance"]["max_peak_vram_gib"] == 9.0
