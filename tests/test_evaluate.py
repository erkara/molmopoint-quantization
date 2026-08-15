import tomllib
from pathlib import Path

import pytest

from evaluate import (
    OFFICIAL_MOLMO2_COMMIT,
    default_run_name,
    parse_variants,
    resolve_image_path,
    select_positions,
    summarize_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_official_extra_declares_direct_http_dependencies():
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    official = project["project"]["optional-dependencies"]["official"]
    assert "httpx>=0.27,<1" in official
    assert "openai>=1,<3" in official


def test_resolve_image_path_rebases_path_after_repository_moves(tmp_path):
    project = tmp_path / "new-project"
    image = project / "data/pixmo-points-eval/images/ab/example.img"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    stale = Path(
        "/home/erdi/Dropbox/Docs/GitRepos/Molmo-Quantization/"
        "data/pixmo-points-eval/images/ab/example.img"
    )

    resolved = resolve_image_path(
        stale,
        data_dir=project / "data/official",
        project_dir=project,
    )

    assert resolved == image


def test_resolve_image_path_prefers_an_existing_original(tmp_path):
    original = tmp_path / "original.img"
    original.write_bytes(b"image")

    resolved = resolve_image_path(
        original,
        data_dir=tmp_path / "data",
        project_dir=tmp_path / "project",
    )

    assert resolved == original


def test_resolve_image_path_lists_attempted_locations(tmp_path):
    stale = Path("/former/project/data/missing/image.img")

    with pytest.raises(FileNotFoundError) as error:
        resolve_image_path(
            stale,
            data_dir=tmp_path / "project/data/official",
            project_dir=tmp_path / "project",
        )

    message = str(error.value)
    assert "Attempted locations:" in message
    assert str(stale) in message
    assert str(tmp_path / "project/data/missing/image.img") in message


def test_failed_examples_make_evaluation_incomplete():
    records = [
        _record(0, metrics={"precision": 1.0, "recall": 1.0, "f1": 1.0}),
        {"source_index": 1, "status": "error"},
    ]
    summary = summarize_records(
        records,
        task="pointing_eval_v2",
        variant="int4",
        model="final-int4",
        official_commit=OFFICIAL_MOLMO2_COMMIT,
        expected_examples=2,
        public_source_examples=2,
        selected_examples=2,
        max_new_tokens=192,
    )

    assert summary["passed_examples"] == 1
    assert summary["failed_examples"] == 1
    assert summary["evaluation_complete"] is False
