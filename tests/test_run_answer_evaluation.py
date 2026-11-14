"""Tests for the GraphRAG answer-quality evaluation runner."""

import json

import pytest

from src.answer_evaluator import (
    AnswerQualityEvaluation,
    JudgeDimensionResult,
)
from src.run_answer_evaluation import (
    evaluate_answers,
    format_evidence_context,
    load_golden_dataset,
    run_answer_evaluation,
)


def make_sample():
    return {
        "id": "RET-001",
        "query": "What depends on RedisCache?",
        "expected_doc_ids": ["DOC-001"],
        "expected_graph_entities": ["AuthService", "RedisCache"],
    }


def make_dataset_file(tmp_path, dataset=None):
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps([make_sample()] if dataset is None else dataset),
        encoding="utf-8",
    )
    return path


class MockPipeline:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.calls = []

    def run(self, query, top_k=3, hops=2):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "hops": hops,
            }
        )

        if self.should_fail:
            raise RuntimeError("Simulated generation failure")

        return {
            "answer": "AuthService depends on RedisCache.",
            "retrieved_chunks": [
                {
                    "id": "DOC-001",
                    "title": "Authentication",
                    "category": "Architecture",
                    "content": (
                        "AuthService depends on RedisCache for session information."
                    ),
                }
            ],
            "graph_context": ["AuthService --[DEPENDS_ON]--> RedisCache"],
            "graph_entities_used": [
                "AuthService",
                "RedisCache",
            ],
            "latency_ms": 12.5,
        }


class MockEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate_answer(self, query, answer, context):
        self.calls.append(
            {
                "query": query,
                "answer": answer,
                "context": context,
            }
        )

        dimension = JudgeDimensionResult(
            score=4.0,
            explanation="Supported by the supplied evidence.",
            supporting_evidence=["AuthService depends on RedisCache."],
            unsupported_claims=[],
        )

        relevance = JudgeDimensionResult(
            score=5.0,
            explanation="Directly answers the question.",
        )

        return AnswerQualityEvaluation(
            query=query,
            faithfulness=dimension,
            answer_relevance=relevance,
            overall_score=4.5,
            evaluation_mode="mock",
        )


def test_load_golden_dataset_success(tmp_path):
    path = make_dataset_file(tmp_path)

    result = load_golden_dataset(path)

    assert len(result) == 1
    assert result[0]["id"] == "RET-001"


def test_load_golden_dataset_limit(tmp_path):
    dataset = [
        make_sample(),
        {**make_sample(), "id": "RET-002"},
    ]

    path = make_dataset_file(tmp_path, dataset)

    result = load_golden_dataset(path, limit=1)

    assert len(result) == 1


def test_load_golden_dataset_rejects_duplicate_ids(tmp_path):
    sample = make_sample()
    path = make_dataset_file(tmp_path, [sample, sample])

    with pytest.raises(ValueError, match="Duplicate sample ID"):
        load_golden_dataset(path)


def test_load_golden_dataset_rejects_missing_fields(tmp_path):
    path = make_dataset_file(
        tmp_path,
        [{"id": "RET-001", "query": "Test?"}],
    )

    with pytest.raises(ValueError, match="missing fields"):
        load_golden_dataset(path)


def test_load_golden_dataset_rejects_empty_dataset(tmp_path):
    path = make_dataset_file(tmp_path, [])

    with pytest.raises(ValueError, match="non-empty"):
        load_golden_dataset(path)


def test_load_golden_dataset_rejects_invalid_limit(tmp_path):
    path = make_dataset_file(tmp_path)

    with pytest.raises(ValueError, match="positive integer"):
        load_golden_dataset(path, limit=0)


def test_format_evidence_context_contains_documents_and_graph():
    context = format_evidence_context(
        {
            "retrieved_chunks": [
                {
                    "id": "DOC-001",
                    "title": "Authentication",
                    "category": "Architecture",
                    "content": "AuthService depends on RedisCache.",
                }
            ],
            "graph_context": ["AuthService --[DEPENDS_ON]--> RedisCache"],
        }
    )

    assert "[Retrieved Documents]" in context
    assert "DOC-001" in context
    assert "Authentication" in context
    assert "[Knowledge Graph Relations]" in context
    assert "DEPENDS_ON" in context


def test_format_context_handles_empty_evidence():
    context = format_evidence_context(
        {
            "retrieved_chunks": [],
            "graph_context": [],
        }
    )

    assert "No relevant documents" in context
    assert "No graph relationships" in context


def test_evaluate_answers_success():
    pipeline = MockPipeline()
    evaluator = MockEvaluator()

    result = evaluate_answers(
        dataset=[make_sample()],
        pipeline=pipeline,
        evaluator=evaluator,
        top_k=5,
        hops=2,
    )

    assert result["sample_count"] == 1
    assert result["successful_samples"] == 1
    assert result["failed_samples"] == 0

    assert result["summary"]["average_faithfulness"] == 4.0
    assert result["summary"]["average_answer_relevance"] == 5.0
    assert result["summary"]["average_overall_score"] == 4.5
    assert result["summary"]["average_generation_latency_ms"] == 12.5

    assert pipeline.calls[0]["top_k"] == 5
    assert len(evaluator.calls) == 1
    assert "DEPENDS_ON" in evaluator.calls[0]["context"]


def test_evaluate_answers_records_generation_failure():
    result = evaluate_answers(
        dataset=[make_sample()],
        pipeline=MockPipeline(should_fail=True),
        evaluator=MockEvaluator(),
    )

    assert result["successful_samples"] == 0
    assert result["failed_samples"] == 1
    assert result["per_sample_results"][0]["status"] == "failed"
    assert result["summary"]["average_overall_score"] is None


def test_evaluate_answers_continues_after_a_failed_sample():
    second_sample = {**make_sample(), "id": "RET-002"}

    class PartiallyFailingPipeline(MockPipeline):
        def run(self, query, top_k=3, hops=2):
            if "fail" in query.lower():
                raise RuntimeError("Simulated failure")
            return super().run(query, top_k, hops)

    dataset = [
        {**make_sample(), "query": "Please fail this query"},
        second_sample,
    ]

    result = evaluate_answers(
        dataset=dataset,
        pipeline=PartiallyFailingPipeline(),
        evaluator=MockEvaluator(),
    )

    assert result["successful_samples"] == 1
    assert result["failed_samples"] == 1


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_evaluate_answers_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        evaluate_answers(
            [make_sample()],
            MockPipeline(),
            MockEvaluator(),
            top_k=top_k,
        )


@pytest.mark.parametrize("hops", [-1, True])
def test_evaluate_answers_rejects_invalid_hops(hops):
    with pytest.raises(ValueError, match="hops"):
        evaluate_answers(
            [make_sample()],
            MockPipeline(),
            MockEvaluator(),
            hops=hops,
        )


def test_evaluate_answers_rejects_empty_dataset():
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_answers([], MockPipeline(), MockEvaluator())


def test_run_answer_evaluation_writes_json_report(tmp_path):
    dataset_path = make_dataset_file(tmp_path)
    report_path = tmp_path / "reports" / "answer_report.json"

    report = run_answer_evaluation(
        dataset_path=dataset_path,
        report_path=report_path,
        pipeline=MockPipeline(),
        evaluator=MockEvaluator(),
        judge_mode="mock",
    )

    assert report_path.exists()
    assert report["successful_samples"] == 1
    assert report["metadata"]["judge_mode"] == "mock"

    saved = json.loads(report_path.read_text(encoding="utf-8"))

    assert saved["sample_count"] == 1
    assert saved["summary"]["average_overall_score"] == 4.5


def test_run_answer_evaluation_rejects_invalid_judge_mode(tmp_path):
    dataset_path = make_dataset_file(tmp_path)

    with pytest.raises(ValueError, match="judge_mode"):
        run_answer_evaluation(
            dataset_path=dataset_path,
            report_path=tmp_path / "report.json",
            judge_mode="invalid",
            pipeline=MockPipeline(),
            evaluator=MockEvaluator(),
        )


def test_run_answer_evaluation_rejects_invalid_retrieval_method(
    tmp_path,
):
    dataset_path = make_dataset_file(tmp_path)

    with pytest.raises(ValueError, match="retrieval_method"):
        run_answer_evaluation(
            dataset_path=dataset_path,
            report_path=tmp_path / "report.json",
            retrieval_method="invalid",
            pipeline=MockPipeline(),
            evaluator=MockEvaluator(),
        )
