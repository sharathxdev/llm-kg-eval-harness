import json

import pytest

from src import run_retrieval_evaluation as runner


def test_load_valid_golden_dataset(tmp_path):
    dataset_path = tmp_path / "golden.json"
    dataset = [
        {
            "id": "RET-001",
            "query": "PaymentGateway dependencies",
            "expected_doc_ids": ["DOC-001"],
            "expected_graph_entities": ["PaymentGateway"],
        }
    ]
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    result = runner.load_golden_dataset(dataset_path)

    assert result == dataset


def test_missing_golden_dataset_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        runner.load_golden_dataset(tmp_path / "missing.json")


def test_empty_golden_dataset_raises_error(tmp_path):
    dataset_path = tmp_path / "golden.json"
    dataset_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty"):
        runner.load_golden_dataset(dataset_path)


def test_duplicate_golden_sample_ids_raise_error(tmp_path):
    dataset_path = tmp_path / "golden.json"
    sample = {
        "id": "RET-001",
        "query": "PaymentGateway dependencies",
        "expected_doc_ids": ["DOC-001"],
        "expected_graph_entities": ["PaymentGateway"],
    }
    dataset_path.write_text(
        json.dumps([sample, sample]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        runner.load_golden_dataset(dataset_path)


def test_invalid_hops_raise_error(monkeypatch):
    def unexpected_dataset_load():
        pytest.fail("Dataset should not load for invalid hops.")

    monkeypatch.setattr(runner, "load_golden_dataset", unexpected_dataset_load)

    with pytest.raises(ValueError, match="non-negative integer"):
        runner.run_retrieval_evaluation(hops=-1)


@pytest.mark.parametrize(
    (
        "retrieval_method",
        "expected_report_name",
        "expected_embedding_provider",
    ),
    [
        ("tfidf", "retrieval_report_k3.json", "TF-IDF"),
        ("bm25", "retrieval_report_bm25_k3.json", None),
    ],
)
def test_runner_generates_retrieval_report(
    tmp_path,
    monkeypatch,
    retrieval_method,
    expected_report_name,
    expected_embedding_provider,
):
    dataset = [
        {
            "id": "RET-001",
            "query": "PaymentGateway dependencies",
            "expected_doc_ids": ["DOC-001"],
            "expected_graph_entities": ["PaymentGateway"],
        }
    ]

    monkeypatch.setattr(
        runner,
        "load_golden_dataset",
        lambda: dataset,
    )

    monkeypatch.setattr(
        runner,
        "DOCUMENTS_PATH",
        tmp_path / "documents.json",
    )

    monkeypatch.setattr(
        runner,
        "REPORT_PATH",
        tmp_path / "reports" / "retrieval_report.json",
    )

    class FakeGraph:
        def number_of_edges(self):
            return 2

    class FakeGraphEngine:
        graph = FakeGraph()

    class FakeRetriever:
        def __init__(self, documents_path, embedding_provider, retrieval_method):
            self.documents = [{"id": "DOC-001"}]
            self.entities = [{"name": "PaymentGateway"}]
            self.graph_engine = FakeGraphEngine()

        def retrieve(self, query, top_k, hops):
            return {
                "retrieved_chunks": [
                    {"id": "DOC-001"}
                ],
                "graph_entities": [
                    {"name": "PaymentGateway"}
                ],
            }

    monkeypatch.setattr(
        runner,
        "HybridGraphRetriever",
        FakeRetriever,
    )

    report = runner.run_retrieval_evaluation(
        k=3,
        hops=2,
        retrieval_method=retrieval_method,
    )

    saved_report = json.loads(
        (
            tmp_path / "reports" / expected_report_name
        ).read_text(encoding="utf-8")
    )

    assert report == saved_report
    assert report["configuration"]["query_count"] == 1
    assert report["summary"]["document_retrieval"]["recall_at_3"] == 1.0
    assert report["summary"]["graph_entities"]["f1_score"] == 1.0
    assert (
        report["metadata"]["embedding_provider"]
        == expected_embedding_provider
    )
    expected_method = "TF-IDF" if retrieval_method == "tfidf" else "BM25"
    assert report["metadata"]["retrieval_method"] == expected_method
    assert report["metadata"]["document_count"] == 1
    assert report["metadata"]["graph_relationship_count"] == 2


def test_invalid_retrieval_method_raises_error(monkeypatch):
    def unexpected_dataset_load():
        pytest.fail("Dataset should not load for an invalid retrieval method.")

    monkeypatch.setattr(
        runner,
        "load_golden_dataset",
        unexpected_dataset_load,
    )

    with pytest.raises(ValueError, match="retrieval_method"):
        runner.run_retrieval_evaluation(retrieval_method="invalid")
