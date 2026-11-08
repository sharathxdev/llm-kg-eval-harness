"""Graph-augmented LLM generation pipeline."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

from src.retriever import HybridGraphRetriever

logger = logging.getLogger(__name__)

load_dotenv()


class LLMGenerationError(RuntimeError):
    """Raised when the LLM provider cannot generate an answer."""


class GraphRAGPipeline:
    """
    Generate grounded answers using retrieved documents and graph facts.
    """

    SYSTEM_PROMPT = """
You are an enterprise technical assistant.

Answer the user's question using only the evidence provided.

Rules:
1. Use retrieved documents as documentary evidence.
2. Use knowledge graph relationships as explicit relational evidence.
3. Never invent services, dependencies, incidents, or relationships.
4. Distinguish documented facts from graph relationships.
5. If the evidence is insufficient, explicitly say so.
6. Do not claim causation unless the evidence establishes causation.
7. Give clear, technically accurate answers.
"""

    def __init__(
        self,
        retriever: HybridGraphRetriever,
        client: Any = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> None:
        self.retriever = retriever
        self.client = client
        self.model = model or os.getenv(
            "OPENAI_MODEL",
            "gpt-4o-mini",
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _get_client(self) -> Any:
        """Create an OpenAI client only when generation is requested."""
        if self.client is not None:
            return self.client

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise LLMGenerationError(
                "OPENAI_API_KEY is not configured."
            )

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key)
            return self.client

        except Exception as exc:
            raise LLMGenerationError(
                f"Unable to initialize the LLM client: {exc}"
            ) from exc

    def _build_user_prompt(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        graph_context: list[str],
    ) -> str:
        """Separate document evidence from graph relationship evidence."""

        document_section = "\n".join(
            [
                (
                    f"Document ID: {chunk['id']}\n"
                    f"Title: {chunk['title']}\n"
                    f"Content: {chunk['content']}"
                )
                for chunk in retrieved_chunks
            ]
        )

        graph_section = "\n".join(graph_context)

        if not document_section:
            document_section = "No relevant documents retrieved."

        if not graph_section:
            graph_section = "No graph relationships retrieved."

        return f"""
[User Question]
{query}

[Retrieved Documents]
{document_section}

[Knowledge Graph Relations]
{graph_section}

Answer using only the available evidence.
If evidence is insufficient, state that clearly.
"""

    def run(
        self,
        query: str,
        hops: int = 2,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """
        Retrieve evidence and generate an LLM answer.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        if hops < 0:
            raise ValueError("hops cannot be negative.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        start_time = time.perf_counter()

        retrieval = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            hops=hops,
        )

        retrieved_chunks = retrieval["retrieved_chunks"]
        graph_context = retrieval["graph_context"]

        graph_entities = [
            entity["name"]
            for entity in retrieval["graph_entities"]
        ]

        if not retrieved_chunks and not graph_context:
            answer = (
                "Insufficient evidence: no relevant documents "
                "or knowledge graph relationships were found."
            )

            return {
                "query": query,
                "answer": answer,
                "retrieved_chunks": [],
                "graph_entities_used": [],
                "graph_context": [],
                "latency_ms": (
                    time.perf_counter() - start_time
                ) * 1000,
            }

        user_prompt = self._build_user_prompt(
            query,
            retrieved_chunks,
            graph_context,
        )

        try:
            client = self._get_client()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content

            if not answer:
                raise LLMGenerationError(
                    "The LLM returned an empty response."
                )

        except LLMGenerationError:
            raise

        except Exception as exc:
            logger.exception("LLM generation failed.")

            raise LLMGenerationError(
                f"LLM generation failed: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "query": query,
            "answer": answer.strip(),
            "retrieved_chunks": retrieved_chunks,
            "graph_entities_used": graph_entities,
            "graph_context": graph_context,
            "latency_ms": round(latency_ms, 3),
        }
