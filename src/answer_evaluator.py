"""LLM-as-a-Judge evaluation for GraphRAG answers."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import get_openai_api_key, get_openai_model
from src.judge_prompts import (
    ANSWER_RELEVANCE_PROMPT,
    FAITHFULNESS_PROMPT,
    JUDGE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class JudgeDimensionResult(BaseModel):
    """Evaluation result for one quality dimension."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=1.0, le=5.0)
    explanation: str = Field(min_length=1)
    supporting_evidence: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)


class AnswerQualityEvaluation(BaseModel):
    """Combined answer quality evaluation."""

    model_config = ConfigDict(extra="forbid")

    query: str
    faithfulness: JudgeDimensionResult
    answer_relevance: JudgeDimensionResult
    overall_score: float = Field(ge=1.0, le=5.0)
    evaluation_mode: Literal["llm", "mock"]
    model: str | None = None


class LLMJudgeEvaluator:
    """Evaluate answer faithfulness and relevance."""

    STOPWORDS: ClassVar[set[str]] = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "into",
        "not",
        "only",
        "using",
        "based",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "about",
        "its",
        "answer",
        "please",
    }

    def __init__(
        self,
        client: Any = None,
        model: str | None = None,
        mock_mode: bool = False,
    ) -> None:
        if not isinstance(mock_mode, bool):
            raise TypeError("mock_mode must be a boolean.")

        self.client = client
        self.model = get_openai_model(model)
        self.mock_mode = mock_mode

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        tokens = re.findall(
            r"[a-z0-9]+(?:-[a-z0-9]+)*",
            text.casefold(),
        )
        return {
            token for token in tokens if len(token) > 2 and token not in cls.STOPWORDS
        }

    @staticmethod
    def _validate_inputs(
        query: str,
        answer: str,
        context: str,
    ) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must not be empty.")

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answer must not be empty.")

        if not isinstance(context, str):
            raise TypeError("Context must be a string.")

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client

        api_key = get_openai_api_key()

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. "
                "Configure it or explicitly enable mock_mode."
            )

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key)
            return self.client
        except Exception as exc:
            raise RuntimeError(f"Unable to initialize judge client: {exc}") from exc

    @staticmethod
    def _build_prompt(
        rubric: str,
        query: str,
        answer: str,
        context: str,
    ) -> str:
        payload = {
            "query": query,
            "answer": answer,
            "context": context,
        }

        return (
            f"{rubric}\n\n"
            "Evaluate this JSON data:\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    def _request_judge(self, prompt: str) -> str:
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": JUDGE_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            if not isinstance(content, str) or not content.strip():
                raise ValueError("Judge returned an empty response.")

            return content.strip()

        except Exception as exc:
            logger.exception("Judge API request failed.")
            raise RuntimeError(f"Judge API request failed: {exc}") from exc

    @staticmethod
    def _parse_result(content: str) -> JudgeDimensionResult:
        cleaned = content.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError("Judge returned invalid JSON.") from exc

        if not isinstance(data, dict):
            raise TypeError("Judge response must be a JSON object.")

        return JudgeDimensionResult.model_validate(data)

    @staticmethod
    def _score_from_overlap(ratio: float) -> int:
        if ratio >= 0.85:
            return 5
        if ratio >= 0.65:
            return 4
        if ratio >= 0.40:
            return 3
        if ratio >= 0.20:
            return 2
        return 1

    def _mock_evaluate(
        self,
        dimension: Literal["faithfulness", "answer_relevance"],
        query: str,
        answer: str,
        context: str,
    ) -> JudgeDimensionResult:
        if dimension == "faithfulness":
            candidate = self._tokens(answer)
            reference = self._tokens(context)
            description = "answer tokens supported by context"
        else:
            candidate = self._tokens(query)
            reference = self._tokens(answer)
            description = "question tokens covered by answer"

        denominator = len(candidate)
        matches = len(candidate & reference)
        ratio = matches / denominator if denominator else 0.0

        score = self._score_from_overlap(ratio)

        return JudgeDimensionResult(
            score=float(score),
            explanation=(
                "MOCK HEURISTIC ONLY - not a real LLM judgment. "
                f"{matches}/{denominator} {description} matched "
                f"({ratio:.1%} overlap)."
            ),
        )

    def _evaluate_dimension(
        self,
        dimension: Literal["faithfulness", "answer_relevance"],
        query: str,
        answer: str,
        context: str,
    ) -> JudgeDimensionResult:
        self._validate_inputs(query, answer, context)

        if self.mock_mode:
            return self._mock_evaluate(dimension, query, answer, context)

        rubric = (
            FAITHFULNESS_PROMPT
            if dimension == "faithfulness"
            else ANSWER_RELEVANCE_PROMPT
        )

        prompt = self._build_prompt(rubric, query, answer, context)

        result = self._parse_result(self._request_judge(prompt))

        logger.info(
            "%s score: %.2f",
            dimension,
            result.score,
        )

        return result

    def evaluate_faithfulness(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> JudgeDimensionResult:
        return self._evaluate_dimension("faithfulness", query, answer, context)

    def evaluate_answer_relevance(
        self,
        query: str,
        answer: str,
        context: str = "",
    ) -> JudgeDimensionResult:
        return self._evaluate_dimension("answer_relevance", query, answer, context)

    def evaluate_answer(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> AnswerQualityEvaluation:
        self._validate_inputs(query, answer, context)

        faithfulness = self.evaluate_faithfulness(query, answer, context)

        relevance = self.evaluate_answer_relevance(query, answer, context)

        overall = round(
            (faithfulness.score + relevance.score) / 2,
            2,
        )

        return AnswerQualityEvaluation(
            query=query,
            faithfulness=faithfulness,
            answer_relevance=relevance,
            overall_score=overall,
            evaluation_mode="mock" if self.mock_mode else "llm",
            model=None if self.mock_mode else self.model,
        )
