"""Tests for judging previously generated GraphRAG answers."""

import json

import pytest

from src.run_saved_answer_judging import (
    judge_saved_report,
    load_answer_report,
)


def make_report():
    return {
        "metadata": {
            "retrieval_method": "tfidf",
            "top_k": 3,
            "hops": 2,
            "judge_mode": "mock",
        },
        "sample_count": 2,
        "successful_samples": 1,
        "failed_samples": 1,
        "summary": {
            "average_faithfulness": 3.0,
            "average_answer_relevance": 4.0,
            "average_overall_score": 3.5,
            "average_generation_latency_ms": 5000.0,
        },
        "per_sample_results": [
            {
                "sample_id": "RET-001",
                "status": "success",
                "query": "What depends on RedisCache?",
                "answer": "AuthService depends on RedisCache.",
            },
            {
                "sample_id": "RET-002",
                "status": "failed",
                "error": "Generation failed.",
            },
        ],
    }


def write_report(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(make_report()), encoding="utf-8")
    return path


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, top_k=3, hops=2):
        self.calls.append((query, top_k, hops))

        return {
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


class FakeEvaluator:
    def __init__(self, should_fail=False):
        self.calls = []
        self.should_fail = should_fail

    def evaluate_answer(self, query, answer, context):
        self.calls.append((query, answer, context))

        if self.should_fail:
            raise RuntimeError("Simulated judge failure")

        return {
            "query": query,
            "faithfulness": {
                "score": 5.0,
                "explanation": "All claims are supported.",
                "supporting_evidence": [],
                "unsupported_claims": [],
            },
            "answer_relevance": {
                "score": 4.0,
                "explanation": "Directly relevant.",
                "supporting_evidence": [],
                "unsupported_claims": [],
            },
            "overall_score": 4.5,
            "evaluation_mode": "llm",
            "model": "test-model",
        }


def test_load_answer_report_success(tmp_path):
    path = write_report(tmp_path)

    report = load_answer_report(path)

    assert report["sample_count"] == 2


def test_load_answer_report_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_answer_report(tmp_path / "missing.json")


def test_load_answer_report_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        load_answer_report(path)


def test_load_answer_report_without_successful_answers(tmp_path):
    report = make_report()
    report["per_sample_results"] = [{"sample_id": "RET-001", "status": "failed"}]

    path = tmp_path / "failed.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="No successfully generated"):
        load_answer_report(path)


def test_judge_saved_report_preserves_original(tmp_path):
    input_path = write_report(tmp_path)
    output_path = tmp_path / "output.json"

    original_content = input_path.read_text(encoding="utf-8")

    retriever = FakeRetriever()
    evaluator = FakeEvaluator()

    report = judge_saved_report(
        input_path=input_path,
        output_path=output_path,
        retriever=retriever,
        evaluator=evaluator,
    )

    assert output_path.exists()
    assert input_path.read_text(encoding="utf-8") == original_content

    assert report["judge_successful_samples"] == 1
    assert report["judge_failed_samples"] == 0

    assert report["summary"]["average_faithfulness"] == 5.0
    assert report["summary"]["average_answer_relevance"] == 4.0
    assert report["summary"]["average_overall_score"] == 4.5

    assert report["metadata"]["judge_mode"] == "llm"
    assert report["per_sample_results"][0]["judge_status"] == "success"
    assert report["per_sample_results"][1]["judge_status"] == "skipped"

    assert len(retriever.calls) == 1
    assert len(evaluator.calls) == 1


def test_judge_uses_saved_context_without_retrieving_again(tmp_path):
    report = make_report()

    exact_context = (
        "[Retrieved Documents]\\n"
        "Original generation evidence DOC-001.\\n"
        "[Knowledge Graph Relations]\\n"
        "AuthService --[DEPENDS_ON]--> RedisCache"
    )

    report["per_sample_results"][0]["evaluation_context"] = exact_context

    input_path = tmp_path / "input_with_context.json"
    output_path = tmp_path / "output.json"

    input_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    retriever = FakeRetriever()
    evaluator = FakeEvaluator()

    judge_saved_report(
        input_path=input_path,
        output_path=output_path,
        retriever=retriever,
        evaluator=evaluator,
    )

    assert retriever.calls == []
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0][2] == exact_context


def test_judge_saved_report_records_judge_failure(tmp_path):
    input_path = write_report(tmp_path)
    output_path = tmp_path / "output.json"

    report = judge_saved_report(
        input_path=input_path,
        output_path=output_path,
        retriever=FakeRetriever(),
        evaluator=FakeEvaluator(should_fail=True),
    )

    assert report["judge_successful_samples"] == 0
    assert report["judge_failed_samples"] == 1

    sample = report["per_sample_results"][0]

    assert sample["judge_status"] == "failed"
    assert "Simulated judge failure" in sample["judge_error"]


def test_input_and_output_paths_must_differ(tmp_path):
    path = write_report(tmp_path)

    with pytest.raises(ValueError, match="must differ"):
        judge_saved_report(
            input_path=path,
            output_path=path,
            retriever=FakeRetriever(),
            evaluator=FakeEvaluator(),
        )


def test_invalid_saved_retrieval_method(tmp_path):
    path = write_report(tmp_path)
    report = make_report()
    report["metadata"]["retrieval_method"] = "invalid"

    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="retrieval method"):
        judge_saved_report(
            input_path=path,
            output_path=tmp_path / "output.json",
            retriever=FakeRetriever(),
            evaluator=FakeEvaluator(),
        )
