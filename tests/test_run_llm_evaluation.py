"""Tests for the real LLM evaluation runner."""

import json
from types import SimpleNamespace

import pytest

from src import run_llm_evaluation as runner


def make_golden_sample():
    return {
        "id": "EVAL-001",
        "text": "AuthService authenticates incoming requests.",
        "expected_entities": ["AuthService"],
        "expected_relations": [],
    }


def make_extraction():
    return SimpleNamespace(
        entities=[
            SimpleNamespace(name="AuthService")
        ],
        relations=[],
    )


def test_load_golden_dataset_success(tmp_path, monkeypatch):
    dataset = [make_golden_sample()]
    dataset_path = tmp_path / "golden_dataset.json"

    dataset_path.write_text(
        json.dumps(dataset),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runner,
        "GOLDEN_DATASET_PATH",
        dataset_path,
    )

    result = runner.load_golden_dataset()

    assert result == dataset
    assert len(result) == 1


def test_load_golden_dataset_rejects_duplicate_ids(
    tmp_path,
    monkeypatch,
):
    sample = make_golden_sample()
    dataset_path = tmp_path / "golden_dataset.json"

    dataset_path.write_text(
        json.dumps([sample, sample]),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runner,
        "GOLDEN_DATASET_PATH",
        dataset_path,
    )

    with pytest.raises(ValueError, match="Duplicate sample ID"):
        runner.load_golden_dataset()


def test_convert_extraction_to_predictions():
    extraction = make_extraction()

    entities, relations = (
        runner.convert_extraction_to_predictions(extraction)
    )

    assert entities == ["AuthService"]
    assert relations == []


def test_evaluate_real_predictions_with_mock_extractor():
    class MockExtractor:
        last_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "total_tokens": 130,
            "latency_seconds": 0.5,
        }

        def extract(self, document):
            assert document["id"] == "EVAL-001"
            return make_extraction()

    results = runner.evaluate_real_predictions(
        dataset=[make_golden_sample()],
        extractor=MockExtractor(),
    )

    assert results["sample_count"] == 1
    assert results["successful_samples"] == 1
    assert results["failed_samples"] == 0
    assert results["metrics"]["entities"]["f1_score"] == 1.0

    assert results["performance"]["total_prompt_tokens"] == 100
    assert results["performance"]["total_completion_tokens"] == 30
    assert results["performance"]["total_tokens"] == 130

    assert results["sample_execution_details"][0]["status"] == "success"


def test_evaluate_real_predictions_records_failed_sample():
    class MockExtractor:
        last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
        }

        def extract(self, document):
            raise RuntimeError("Simulated API failure")

    with pytest.raises(
        RuntimeError,
        match="No samples were successfully extracted",
    ):
        runner.evaluate_real_predictions(
            dataset=[make_golden_sample()],
            extractor=MockExtractor(),
        )


def test_create_report_records_actual_model_name():
    report = runner.create_report(
        results={
            "sample_count": 1,
            "successful_samples": 1,
            "failed_samples": 0,
            "metrics": {
                "entities": {},
                "relations": {},
                "overall": {},
            },
            "performance": {},
            "per_sample_results": [],
            "sample_execution_details": [],
        },
        model_name="gpt-4o-mini",
    )

    assert report["model"] == "gpt-4o-mini"
    assert report["evaluation_type"] == "real_llm_extraction"


def test_select_golden_samples_success_preserves_dataset_order():
    dataset = [
        {"id": "EVAL-001"},
        {"id": "EVAL-002"},
        {"id": "EVAL-003"},
        {"id": "EVAL-004"},
    ]

    selected = runner.select_golden_samples(
        dataset,
        ["EVAL-004", "EVAL-002"],
    )

    assert [sample["id"] for sample in selected] == [
        "EVAL-002",
        "EVAL-004",
    ]


def test_select_golden_samples_rejects_unknown_ids():
    dataset = [
        {"id": "EVAL-001"},
        {"id": "EVAL-002"},
    ]

    with pytest.raises(ValueError, match="Unknown sample IDs: EVAL-999"):
        runner.select_golden_samples(dataset, ["EVAL-999"])


def test_select_golden_samples_rejects_duplicate_ids():
    dataset = [
        {"id": "EVAL-001"},
        {"id": "EVAL-002"},
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate sample IDs were provided",
    ):
        runner.select_golden_samples(
            dataset,
            ["EVAL-001", "EVAL-001"],
        )


def test_select_golden_samples_rejects_empty_selection():
    dataset = [
        {"id": "EVAL-001"},
        {"id": "EVAL-002"},
    ]

    with pytest.raises(
        ValueError,
        match="At least one sample ID must be provided",
    ):
        runner.select_golden_samples(dataset, [])


def test_run_evaluation_targeted_selection_preserves_baseline(
    tmp_path,
    monkeypatch,
):
    dataset = [
        {"id": "EVAL-001"},
        {"id": "EVAL-002"},
        {"id": "EVAL-003"},
    ]

    baseline_path = tmp_path / "llm_evaluation_report.json"
    targeted_path = tmp_path / "llm_evaluation_report_targeted.json"

    baseline_content = '{"original": "baseline"}'
    baseline_path.write_text(baseline_content, encoding="utf-8")

    captured = {}

    class MockExtractor:
        model = "mock-model"

    def mock_evaluate_real_predictions(dataset, extractor):
        captured["ids"] = [sample["id"] for sample in dataset]
        return {"selected_ids": captured["ids"]}

    monkeypatch.setattr(runner, "load_golden_dataset", lambda: dataset)
    monkeypatch.setattr(
        runner,
        "LLMEntityExtractor",
        MockExtractor,
    )
    monkeypatch.setattr(
        runner,
        "evaluate_real_predictions",
        mock_evaluate_real_predictions,
    )
    monkeypatch.setattr(
        runner,
        "create_report",
        lambda results, model_name: {
            "model": model_name,
            "selected_ids": results["selected_ids"],
        },
    )
    monkeypatch.setattr(runner, "print_summary", lambda report: None)
    monkeypatch.setattr(runner, "REPORT_DIRECTORY", tmp_path)
    monkeypatch.setattr(runner, "REPORT_PATH", baseline_path)
    monkeypatch.setattr(
        runner,
        "TARGETED_REPORT_PATH",
        targeted_path,
    )

    report = runner.run_evaluation(
        sample_ids=["EVAL-003", "EVAL-001"],
    )

    assert captured["ids"] == ["EVAL-001", "EVAL-003"]
    assert report["selected_ids"] == ["EVAL-001", "EVAL-003"]
    assert targeted_path.exists()
    assert json.loads(targeted_path.read_text()) == report
    assert baseline_path.read_text() == baseline_content


def test_run_evaluation_limit_uses_original_report_path(
    tmp_path,
    monkeypatch,
):
    dataset = [
        {"id": "EVAL-001"},
        {"id": "EVAL-002"},
        {"id": "EVAL-003"},
    ]

    baseline_path = tmp_path / "llm_evaluation_report.json"
    targeted_path = tmp_path / "llm_evaluation_report_targeted.json"

    captured = {}

    class MockExtractor:
        model = "mock-model"

    def mock_evaluate_real_predictions(dataset, extractor):
        captured["ids"] = [sample["id"] for sample in dataset]
        return {"selected_ids": captured["ids"]}

    monkeypatch.setattr(runner, "load_golden_dataset", lambda: dataset)
    monkeypatch.setattr(
        runner,
        "LLMEntityExtractor",
        MockExtractor,
    )
    monkeypatch.setattr(
        runner,
        "evaluate_real_predictions",
        mock_evaluate_real_predictions,
    )
    monkeypatch.setattr(
        runner,
        "create_report",
        lambda results, model_name: {
            "model": model_name,
            "selected_ids": results["selected_ids"],
        },
    )
    monkeypatch.setattr(runner, "print_summary", lambda report: None)
    monkeypatch.setattr(runner, "REPORT_DIRECTORY", tmp_path)
    monkeypatch.setattr(runner, "REPORT_PATH", baseline_path)
    monkeypatch.setattr(
        runner,
        "TARGETED_REPORT_PATH",
        targeted_path,
    )

    runner.run_evaluation(limit=2)

    assert captured["ids"] == ["EVAL-001", "EVAL-002"]
    assert baseline_path.exists()
    assert not targeted_path.exists()


def test_run_evaluation_rejects_limit_and_sample_ids_together():
    with pytest.raises(
        ValueError,
        match="Use either --limit or --sample-ids, not both",
    ):
        runner.run_evaluation(
            limit=2,
            sample_ids=["EVAL-001"],
        )
