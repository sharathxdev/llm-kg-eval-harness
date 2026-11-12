"""Hybrid semantic and knowledge-graph retrieval."""

from __future__ import annotations

from collections import deque

import logging
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_processor import DataProcessor
from src.graph_builder import KnowledgeGraphBuilder
from src.graph_engine import GraphQueryEngine
from src.graph_loader import load_knowledge_graph

logger = logging.getLogger(__name__)


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize text while preserving technical identifiers."""
    return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.casefold())


class EmbeddingProvider(Protocol):
    """Interface implemented by document and query embedding providers."""

    def fit(self, texts: list[str]) -> None:
        """Prepare an embedding index from document texts."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Convert texts into comparable feature vectors."""


class TfidfEmbeddingProvider:
    """Local lexical vectorizer used when transformer embeddings are unavailable."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.transform(texts)


class SentenceTransformerEmbeddingProvider:
    """Generate dense semantic embeddings using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self.model: Any = None

    def _load_model(self) -> None:
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)

    def fit(self, texts: list[str]) -> None:
        """Sentence Transformer models do not require corpus fitting."""
        self._load_model()

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        return np.asarray(
            self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )


class AutoEmbeddingProvider:
    """Prefer dense transformer embeddings and fall back to local TF-IDF."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name
        self.provider: EmbeddingProvider | None = None

    def fit(self, texts: list[str]) -> None:
        try:
            self.provider = SentenceTransformerEmbeddingProvider(
                self.model_name or "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.provider.fit(texts)
            logger.info("Using Sentence Transformer embeddings.")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Transformer embeddings unavailable; using TF-IDF fallback: %s",
                exc,
            )
            self.provider = TfidfEmbeddingProvider()
            self.provider.fit(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.provider is None:
            raise RuntimeError("Embedding provider has not been fitted.")
        return self.provider.encode(texts)


class HybridGraphRetriever:
    """
    Retrieve relevant documents and knowledge-graph relationships.

    The existing graph engine is reused for multi-hop traversal.
    """

    def __init__(
        self,
        documents_path: str | Path = "data/documents.json",
        graph_engine: GraphQueryEngine | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        documents: list[dict[str, Any]] | None = None,
        retrieval_method: str = "tfidf",
        graph_path: str | Path | None = None,
        rrf_constant: int = 60,
    ) -> None:
        if retrieval_method not in {"tfidf", "bm25", "rrf"}:
            raise ValueError(
                "retrieval_method must be either 'tfidf' or 'bm25' (or 'rrf')."
            )

        if (
            not isinstance(rrf_constant, int)
            or isinstance(rrf_constant, bool)
            or rrf_constant < 1
        ):
            raise ValueError("rrf_constant must be a positive integer.")

        self.retrieval_method = retrieval_method
        self.documents_path = Path(documents_path)
        self.rrf_constant = rrf_constant

        # TF-IDF and RRF require an embedding provider.
        self.embedding_provider = (
            embedding_provider or AutoEmbeddingProvider()
            if retrieval_method in {"tfidf", "rrf"}
            else None
        )

        if documents is None:
            self.documents = DataProcessor().load_documents(
                self.documents_path
            )
        else:
            self.documents = DataProcessor().process_documents(documents)

        if graph_engine is not None and graph_path is not None:
            raise ValueError(
                "Provide either graph_engine or graph_path, not both."
            )

        if graph_engine is not None:
            self.graph_engine = graph_engine
        elif graph_path is not None:
            graph = load_knowledge_graph(graph_path)
            self.graph_engine = GraphQueryEngine(graph)
        else:
            builder = KnowledgeGraphBuilder()
            graph = builder.build_from_corpus(str(self.documents_path))
            self.graph_engine = GraphQueryEngine(graph)

        self.document_texts = [
            f"{doc['title']}. {doc['category']}. {doc['content']}"
            for doc in self.documents
        ]

        self.bm25_corpus = [
            tokenize_bm25(text) for text in self.document_texts
        ]
        self.bm25_index = (
            BM25Okapi(self.bm25_corpus)
            if self.bm25_corpus
            else None
        )
        self.document_embeddings: Any = None

        if self.retrieval_method in {"tfidf", "rrf"} and self.document_texts:
            if self.embedding_provider is None:
                raise RuntimeError(
                    "TF-IDF/RRF retrieval requires an embedding provider."
                )

            self.embedding_provider.fit(self.document_texts)
            self.document_embeddings = self.embedding_provider.encode(
                self.document_texts
            )

        self.entities = self.graph_engine.get_all_entities()

        logger.info(
            "Hybrid retriever initialized with %d documents and %d entities.",
            len(self.documents),
            len(self.entities),
        )

    def _detect_entities(self, query: str) -> list[dict[str, Any]]:
        """Find known graph entity names and IDs in a query."""
        matches = []
        seen: set[str] = set()

        candidates = sorted(
            self.entities,
            key=lambda item: max(
                len(str(item.get("name", ""))),
                len(str(item.get("id", ""))),
            ),
            reverse=True,
        )

        for entity in candidates:
            entity_id = str(entity["id"])
            name = str(entity.get("name", ""))

            terms = [name, entity_id]
            metadata = entity.get("metadata", {})

            if isinstance(metadata, dict):
                aliases = metadata.get("aliases", [])
                if isinstance(aliases, list):
                    terms.extend(
                        str(alias) for alias in aliases if alias
                    )

            for term in terms:
                if not term.strip():
                    continue

                pattern = (
                    r"(?<![A-Za-z0-9_-])"
                    + re.escape(term.strip())
                    + r"(?![A-Za-z0-9_-])"
                )

                if re.search(pattern, query, flags=re.IGNORECASE):
                    if entity_id not in seen:
                        matches.append(entity)
                        seen.add(entity_id)
                    break

        return matches

    def _retrieve_documents(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rank documents using the configured retrieval method."""
        if self.retrieval_method == "bm25":
            return self._retrieve_documents_bm25(query, top_k)

        if self.retrieval_method == "rrf":
            return self._retrieve_documents_rrf(query, top_k)

        return self._retrieve_documents_tfidf(query, top_k)

    def _retrieve_documents_tfidf(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rank documents using embedding cosine similarity."""
        if (
            not self.documents
            or self.document_embeddings is None
            or self.embedding_provider is None
        ):
            return []

        query_embedding = self.embedding_provider.encode([query])

        scores = cosine_similarity(
            query_embedding,
            self.document_embeddings,
        )[0]

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in ranked_indices:
            document = self.documents[int(index)]
            score = float(scores[index])

            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "category": document["category"],
                    "content": document["content"],
                    "similarity_score": score,
                }
            )

        return results

    def _retrieve_documents_rrf(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Fuse embedding and BM25 rankings using Reciprocal Rank Fusion.

        RRF combines document ranks rather than raw scores because cosine
        similarity and BM25 scores use different numerical scales.

        RRF score = sum(1 / (rrf_constant + rank)), with ranks starting at 1.
        """
        if not self.documents:
            return []

        # Obtain candidate rankings independently from both retrievers.
        embedding_results = self._retrieve_documents_tfidf(query, top_k)
        bm25_results = self._retrieve_documents_bm25(query, top_k)

        fused: dict[str, dict[str, Any]] = {}

        rankings = (
            ("embedding", embedding_results, "similarity_score"),
            ("bm25", bm25_results, "retrieval_score"),
        )

        for method, ranking, score_key in rankings:
            for rank, document in enumerate(ranking, start=1):
                score = float(document.get(score_key, 0.0))

                # Skip zero-score documents without changing original ranks.
                # Negative BM25 scores can still have a meaningful ranking.
                if score == 0.0:
                    continue

                document_id = str(document["id"])

                entry = fused.setdefault(
                    document_id,
                    {
                        "document": document,
                        "rrf_score": 0.0,
                        "embedding_rank": None,
                        "bm25_rank": None,
                        "embedding_score": None,
                        "bm25_score": None,
                    },
                )

                entry["rrf_score"] += (
                    1.0 / (self.rrf_constant + rank)
                )

                entry[f"{method}_rank"] = rank
                entry[f"{method}_score"] = score

        if not fused:
            return []

        # Sort by fused score and then document ID for deterministic ties.
        ranked_results = sorted(
            fused.values(),
            key=lambda item: (
                -item["rrf_score"],
                str(item["document"]["id"]),
            ),
        )

        results = []

        for item in ranked_results[:top_k]:
            document = item["document"]
            score = round(float(item["rrf_score"]), 10)

            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "category": document["category"],
                    "content": document["content"],
                    "retrieval_score": score,
                    "retrieval_method": "rrf",
                    "embedding_rank": item["embedding_rank"],
                    "bm25_rank": item["bm25_rank"],
                    "embedding_score": item["embedding_score"],
                    "bm25_score": item["bm25_score"],
                    "rrf_constant": self.rrf_constant,
                }
            )

        return results

    def _retrieve_documents_bm25(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rank documents using BM25 relevance scores."""
        if not self.documents or self.bm25_index is None:
            return []

        query_tokens = tokenize_bm25(query)

        if not query_tokens:
            return []

        scores = self.bm25_index.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in ranked_indices:
            document = self.documents[int(index)]
            score = float(scores[index])

            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "category": document["category"],
                    "content": document["content"],
                    "retrieval_score": score,
                    "retrieval_method": "bm25",
                }
            )

        return results

    def _get_graph_context(
        self,
        entity_ids: str | list[str],
        hops: int,
    ) -> list[str]:
        """Return relevant relationships using multi-source BFS traversal."""
        graph = self.graph_engine.graph

        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]

        seed_ids = sorted({
            entity_id
            for entity_id in entity_ids
            if graph.has_node(entity_id)
        })

        if not seed_ids:
            return []

        queue = deque((entity_id, 0) for entity_id in seed_ids)
        visited = set(seed_ids)
        statements: set[str] = set()

        while queue:
            current_id, depth = queue.popleft()

            if depth >= hops:
                continue

            neighbors = set(graph.successors(current_id))
            neighbors.update(graph.predecessors(current_id))

            for neighbor_id in sorted(neighbors):
                is_new = neighbor_id not in visited

                # Include direct seed relationships and only the edges
                # that discover new nodes at subsequent traversal levels.
                if depth == 0 or is_new:
                    for source, target in (
                        (current_id, neighbor_id),
                        (neighbor_id, current_id),
                    ):
                        edge_groups = graph.get_edge_data(
                            source,
                            target,
                            default={},
                        )

                        for attributes in edge_groups.values():
                            source_name = graph.nodes[source].get(
                                "name", source
                            )
                            target_name = graph.nodes[target].get(
                                "name", target
                            )
                            relation = attributes.get(
                                "relation_type", "RELATED_TO"
                            )

                            statements.add(
                                f"{source_name} --[{relation}]--> {target_name}"
                            )

                if is_new:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))

        return sorted(statements)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        hops: int = 2,
    ) -> dict[str, Any]:
        """
        Retrieve semantic documents and graph relationships for a query.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if hops < 0:
            raise ValueError("hops cannot be negative.")

        query = query.strip()

        retrieved_chunks = self._retrieve_documents(query, top_k)
        matched_entities = self._detect_entities(query)

        seed_entity_ids = [
            entity["id"] for entity in matched_entities
        ]

        graph_context = self._get_graph_context(
            seed_entity_ids,
            hops,
        )

        context_parts = []

        if retrieved_chunks:
            context_parts.append("[Retrieved Documents]")

            for chunk in retrieved_chunks:
                context_parts.append(
                    f"Document: {chunk['title']}\n"
                    f"Content: {chunk['content']}"
                )

        if graph_context:
            context_parts.append("[Knowledge Graph Relations]")
            context_parts.extend(graph_context)

        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "graph_context": graph_context,
            "graph_entities": [
                {
                    "id": entity["id"],
                    "name": entity.get("name", entity["id"]),
                    "entity_type": entity.get("entity_type"),
                }
                for entity in matched_entities
            ],
            "context_text": "\n\n".join(context_parts),
        }
