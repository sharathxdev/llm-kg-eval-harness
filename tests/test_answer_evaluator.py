"""Tests for LLM-as-a-Judge evaluation."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.answer_evaluator import (
    AnswerQualityEvaluation,
    JudgeDimensionResult,
    LLMJudgeEvaluator,
)


def fake_response(score=4):
    return json.dumps(
        {
            "score": score,
            "explanation": "Evaluation based on supplied evidence.",
            "supporting_evidence": ["AuthService depends on RedisCache."],
            "unsupported_claims": [],
        }
    )


def fake_client(responses):
    class Completions:
        def __init__(self):
            self.responses = list(responses)
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            content = self.responses.pop(0)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_valid_judge_result():
    result = JudgeDimensionResult(
        score=4,
        explanation="Mostly supported.",
    )
    assert result.score == 4


@pytest.mark.parametrize("score", [0, 6, -1, 10])
def test_invalid_judge_score(score):
    with pytest.raises(ValidationError):
        JudgeDimensionResult(
            score=score,
            explanation="Invalid.",
        )


def test_mock_evaluation():
    evaluator = LLMJudgeEvaluator(mock_mode=True)

    result = evaluator.evaluate_answer(
        query="AuthService depends on RedisCache",
        answer="AuthService depends on RedisCache.",
        context="AuthService depends on RedisCache for sessions.",
    )

    assert isinstance(result, AnswerQualityEvaluation)
    assert result.evaluation_mode == "mock"
    assert result.model is None
    assert result.faithfulness.score == 5
    assert result.answer_relevance.score == 5
    assert result.overall_score == 5
    assert "MOCK HEURISTIC ONLY" in result.faithfulness.explanation


def test_mock_detects_low_faithfulness():
    evaluator = LLMJudgeEvaluator(mock_mode=True)

    result = evaluator.evaluate_faithfulness(
        "What happened?",
        "Database crashed unexpectedly.",
        "AuthService handles authentication.",
    )

    assert result.score == 1


def test_mock_detects_low_relevance():
    evaluator = LLMJudgeEvaluator(mock_mode=True)

    result = evaluator.evaluate_answer_relevance(
        "Which service depends on RedisCache?",
        "The weather is sunny today.",
    )

    assert result.score == 1


def test_real_judge_faithfulness_with_mock_client():
    client, completions = fake_client([fake_response(4)])

    evaluator = LLMJudgeEvaluator(
        client=client,
        model="test-model",
    )

    result = evaluator.evaluate_faithfulness(
        "What depends on RedisCache?",
        "AuthService depends on RedisCache.",
        "AuthService depends on RedisCache for sessions.",
    )

    assert result.score == 4
    assert len(completions.calls) == 1
    assert completions.calls[0]["temperature"] == 0


def test_both_judge_dimensions():
    client, completions = fake_client(
        [
            fake_response(4),
            fake_response(5),
        ]
    )

    evaluator = LLMJudgeEvaluator(
        client=client,
        model="test-model",
    )

    result = evaluator.evaluate_answer(
        "What depends on RedisCache?",
        "AuthService depends on RedisCache.",
        "AuthService depends on RedisCache for sessions.",
    )

    assert result.faithfulness.score == 4
    assert result.answer_relevance.score == 5
    assert result.overall_score == 4.5
    assert result.evaluation_mode == "llm"
    assert len(completions.calls) == 2


def test_parser_accepts_markdown_json():
    content = """```json
    {
      "score": 5,
      "explanation": "Supported.",
      "supporting_evidence": [],
      "unsupported_claims": []
    }
    ```"""

    result = LLMJudgeEvaluator._parse_result(content)
    assert result.score == 5


def test_parser_rejects_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        LLMJudgeEvaluator._parse_result("not JSON")


def test_parser_rejects_non_object():
    with pytest.raises(TypeError, match="JSON object"):
        LLMJudgeEvaluator._parse_result('["invalid"]')


def test_parser_rejects_out_of_range_score():
    with pytest.raises(ValidationError):
        LLMJudgeEvaluator._parse_result(fake_response(8))


def test_empty_query_is_rejected():
    evaluator = LLMJudgeEvaluator(mock_mode=True)

    with pytest.raises(ValueError):
        evaluator.evaluate_answer("", "Some answer.", "Some context.")


def test_empty_answer_is_rejected():
    evaluator = LLMJudgeEvaluator(mock_mode=True)

    with pytest.raises(ValueError):
        evaluator.evaluate_answer("Some question?", "", "Some context.")


def test_non_string_context_is_rejected():
    evaluator = LLMJudgeEvaluator(mock_mode=True)

    with pytest.raises(TypeError):
        evaluator.evaluate_answer("Some question?", "Some answer.", None)


def test_real_mode_requires_api_key(monkeypatch):
    import src.answer_evaluator as module

    monkeypatch.setattr(
        module,
        "get_openai_api_key",
        lambda: None,
    )

    evaluator = LLMJudgeEvaluator(model="test-model")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        evaluator.evaluate_faithfulness(
            "What happened?",
            "An incident occurred.",
            "An incident occurred.",
        )
