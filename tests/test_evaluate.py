import tomllib
from pathlib import Path

import pytest

from evaluate import (
    OFFICIAL_MOLMO2_COMMIT,
    default_run_name,
    parse_variants,
    render_run_comparison,
    render_sample_predictions,
    resolve_image_path,
    run_json_directory,
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
    assert parse_variants("int4") == ["bf16", "int4"]
    assert parse_variants("int8") == ["bf16", "int8"]
    assert parse_variants("int4,bf16,int4") == ["bf16", "int4"]
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
    assert "matplotlib>=3.8,<4" in official
    assert "openai>=1,<3" in official
    assert "tqdm>=4.66,<5" in official


@pytest.mark.parametrize(
    ("dataset", "variants", "metrics"),
    [
        ("pixmo-points", ("int4",), {"f1": 0.8}),
        ("pointbench", ("bf16", "int4", "int8"), {"average": 0.7}),
    ],
)
def test_run_comparison_plot_supports_each_dataset_and_variant_scope(
    tmp_path, dataset, variants, metrics
):
    summaries = {
        variant: {
            "metrics": metrics,
            "performance": {"max_peak_vram_gib": 8.0},
            "selected_examples": 5,
            "public_source_examples": 231,
            "expected_examples": 436,
        }
        for variant in variants
    }
    output = tmp_path / "full_comparison.png"

    render_run_comparison(summaries, dataset=dataset, output=output)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_new_runs_use_json_subdirectory_but_legacy_runs_resume_in_place(tmp_path):
    new_run = tmp_path / "new"
    assert run_json_directory(new_run) == new_run / "json_summaries"

    legacy_run = tmp_path / "legacy"
    legacy_run.mkdir()
    (legacy_run / "selection.json").write_text("{}", encoding="utf-8")
    assert run_json_directory(legacy_run) == legacy_run


@pytest.mark.parametrize("variants", [("bf16",), ("bf16", "int4")])
def test_sample_prediction_plot_matches_effective_variants(tmp_path, variants):
    from PIL import Image

    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 24), "white").save(image_path)
    dataset = [{"source_index": index, "image": str(image_path)} for index in range(3)]
    json_dir = tmp_path / "json_summaries"
    json_dir.mkdir()
    for variant in variants:
        records = [
            {
                "source_index": index,
                "status": "passed",
                "prompt": f"Point to example {index}",
                "points": [{"x": 8 + index, "y": 10}],
            }
            for index in range(3)
        ]
        (json_dir / f"{variant}.jsonl").write_text(
            "".join(f"{__import__('json').dumps(record)}\n" for record in records),
            encoding="utf-8",
        )
    output = tmp_path / "sample_predictions.png"

    render_sample_predictions(
        dataset,
        positions=[0, 1, 2],
        variants=list(variants),
        json_dir=json_dir,
        data_dir=tmp_path,
        dataset_name="pointbench",
        seed=0,
        output=output,
    )

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_resolve_image_path_rebases_path_after_repository_moves(tmp_path):
    project = tmp_path / "new-project"
    image = project / "data/pixmo-points-eval/images/ab/example.img"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    stale = Path(
        "/former/checkouts/Molmo-Quantization/"
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
