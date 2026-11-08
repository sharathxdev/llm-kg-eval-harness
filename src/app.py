"""FastAPI application for the GraphRAG system."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from src.pipeline import GraphRAGPipeline, LLMGenerationError
from src.retriever import HybridGraphRetriever

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "documents.json"


class QueryRequest(BaseModel):
    """Request body shared by retrieval and generation endpoints."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question or search query.",
    )
    top_k: int = Field(default=3, ge=1, le=20)
    hops: int = Field(default=2, ge=0, le=5)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Reject whitespace-only queries and normalize surrounding spaces."""
        value = value.strip()
        if not value:
            raise ValueError("Query cannot be empty.")
        return value


def create_app(
    retriever: Any = None,
    pipeline: Any = None,
) -> FastAPI:
    """Create the API application, optionally with injected components."""

    application = FastAPI(
        title="GraphRAG Evaluation Harness API",
        description=(
            "REST API for hybrid document retrieval, "
            "knowledge graph context, and grounded LLM generation."
        ),
        version="1.0.0",
    )

    application.state.retriever = retriever
    application.state.pipeline = pipeline

    def get_retriever(request: Request) -> Any:
        """Initialize the hybrid retriever only when first needed."""
        existing = request.app.state.retriever
        if existing is not None:
            return existing

        try:
            instance = HybridGraphRetriever(
                documents_path=DEFAULT_DOCUMENTS_PATH,
            )
            request.app.state.retriever = instance
            return instance
        except Exception as exc:
            logger.exception("Unable to initialize the retriever.")
            raise HTTPException(
                status_code=503,
                detail="Retrieval service is unavailable.",
            ) from exc

    def get_pipeline(request: Request) -> Any:
        """Initialize the GraphRAG pipeline only when first needed."""
        existing = request.app.state.pipeline
        if existing is not None:
            return existing

        instance = GraphRAGPipeline(
            retriever=get_retriever(request),
        )
        request.app.state.pipeline = instance
        return instance

    @application.get("/health")
    def health() -> dict[str, str]:
        """Return the API process health without loading ML components."""
        return {
            "status": "ok",
            "service": "GraphRAG Evaluation Harness API",
        }

    @application.post("/retrieve")
    def retrieve(
        payload: QueryRequest,
        request: Request,
    ) -> dict[str, Any]:
        """Return document matches and knowledge graph evidence."""
        active_retriever = get_retriever(request)

        try:
            return active_retriever.retrieve(
                query=payload.query,
                top_k=payload.top_k,
                hops=payload.hops,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("Document and graph retrieval failed.")
            raise HTTPException(
                status_code=500,
                detail="Retrieval failed due to an internal error.",
            ) from exc

    @application.post("/query")
    def query(
        payload: QueryRequest,
        request: Request,
    ) -> dict[str, Any]:
        """Return an answer grounded in retrieved documents and graph facts."""
        active_pipeline = get_pipeline(request)

        try:
            return active_pipeline.run(
                query=payload.query,
                top_k=payload.top_k,
                hops=payload.hops,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        except LLMGenerationError as exc:
            raise HTTPException(
                status_code=502,
                detail="The language model could not generate an answer.",
            ) from exc
        except Exception as exc:
            logger.exception("GraphRAG query failed.")
            raise HTTPException(
                status_code=500,
                detail="Query failed due to an internal error.",
            ) from exc

    return application


app = create_app()
