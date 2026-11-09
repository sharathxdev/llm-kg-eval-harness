import json

import pytest

from src import retrieval_comparison as comparison


def create_report(k, precision=0.8, recall=0.7, ndcg=0.9):
    return {
        "configuration": {"k": k},
        "summary": {
            "document_retrieval": {
                f"precision_at_{k}": precision,
                f"recall_at_{k}": recall,
                f"hit_rate_at_{k}": 1.0,
                f"mrr_at_{k}": 1.0,
                f"ndcg_at_{k}": ndcg,
            },
            "graph_entities": {
                "f1_score": 1.0,
            },
        },
    }


def test_load_report_success(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(create_report(3)), encoding="utf-8")

    report = comparison.load_report(path)

    assert report["configuration"]["k"] == 3


def test_load_report_missing_file(tmp_path):
    path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        comparison.load_report(path)


def test_load_report_invalid_structure(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"summary": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="summary"):
        comparison.load_report(path)


def test_extract_metrics():
    report = create_report(
        5,
        precision=0.7,
        recall=0.8,
        ndcg=0.88,
    )

    metrics = comparison.extract_metrics(report, 5)

    assert metrics["precision_at_5"] == 0.7
    assert metrics["recall_at_5"] == 0.8
    assert metrics["hit_rate_at_5"] == 1.0
    assert metrics["mrr_at_5"] == 1.0
    assert metrics["ndcg_at_5"] == 0.88
    assert metrics["graph_entity_f1"] == 1.0


def test_extract_metrics_rejects_missing_metric():
    report = create_report(3)
    del report["summary"]["document_retrieval"]["recall_at_3"]

    with pytest.raises(ValueError, match="recall_at_3"):
        comparison.extract_metrics(report, 3)


def test_build_comparison_identifies_winners_and_ties(
    tmp_path,
    monkeypatch,
):
    report_mapping = {
        "TF-IDF": {
            3: "tfidf_k3.json",
            5: "tfidf_k5.json",
            10: "tfidf_k10.json",
        },
        "BM25": {
            3: "bm25_k3.json",
            5: "bm25_k5.json",
            10: "bm25_k10.json",
        },
    }

    report_values = {
        "tfidf_k3.json": create_report(3, 0.8, 0.7, 0.9),
        "bm25_k3.json": create_report(3, 0.8, 0.6, 0.9),
        "tfidf_k5.json": create_report(5, 0.7, 0.8, 0.9),
        "bm25_k5.json": create_report(5, 0.6, 0.7, 0.8),
        "tfidf_k10.json": create_report(10, 0.4, 0.9, 0.9),
        "bm25_k10.json": create_report(10, 0.4, 0.8, 0.9),
    }

    for filename, report in report_values.items():
        (tmp_path / filename).write_text(
            json.dumps(report),
            encoding="utf-8",
        )

    monkeypatch.setattr(comparison, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(
        comparison,
        "BENCHMARK_REPORTS",
        report_mapping,
    )

    result = comparison.build_comparison()

    assert (
        result["comparisons"][3]["precision_at_3"]["winner"]
        == "Tie"
    )
    assert (
        result["comparisons"][3]["recall_at_3"]["winner"]
        == "TF-IDF"
    )
    assert (
        result["comparisons"][5]["ndcg_at_5"]["winner"]
        == "TF-IDF"
    )
    assert (
        result["comparisons"][10]["precision_at_10"]["winner"]
        == "Tie"
    )


def test_save_comparison_report(tmp_path, monkeypatch):
    output_path = tmp_path / "comparison.json"

    monkeypatch.setattr(
        comparison,
        "REPORTS_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        comparison,
        "COMPARISON_REPORT_PATH",
        output_path,
    )

    report = {"metadata": {"comparison_type": "test"}}

    saved_path = comparison.save_comparison_report(report)

    assert saved_path == output_path
    assert output_path.is_file()

    saved_data = json.loads(
        output_path.read_text(encoding="utf-8")
    )
    assert saved_data == report
