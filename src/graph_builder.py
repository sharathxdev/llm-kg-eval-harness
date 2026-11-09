"""Knowledge graph construction from a domain document corpus."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, ClassVar

import networkx as nx

from src.models import EntityNode, EntityType, ExtractionResult, RelationEdge

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Build a directed, multi-edge knowledge graph from JSON documents."""

    ENTITY_TYPES: ClassVar[dict[str, EntityType]] = {
        "AuthService": EntityType.SERVICE,
        "PaymentGateway": EntityType.SERVICE,
        "PostgresCluster": EntityType.DATABASE,
        "RedisCache": EntityType.DATABASE,
        "Incident-304": EntityType.INCIDENT,
        "IAM-Failover-Policy": EntityType.POLICY,
        "NotificationService": EntityType.SERVICE,
        "OrderService": EntityType.SERVICE,
        "API Gateway": EntityType.SERVICE,
        "MonitoringService": EntityType.SERVICE,
        "KafkaCluster": EntityType.DATABASE,
        "InventoryService": EntityType.SERVICE,
        "PlatformEngineering": EntityType.TEAM,
    }

    RELATION_RULES: tuple[tuple[str, str], ...] = (
        ("depends on", "DEPENDS_ON"),
        ("depend on", "DEPENDS_ON"),
        ("communicates with", "WORKS_WITH"),
        ("coordinates with", "WORKS_WITH"),
        ("works with", "WORKS_WITH"),
        ("affecting", "AFFECTS"),
        ("affects", "AFFECTS"),
        ("impacts", "AFFECTS"),
        ("governs", "GOVERNS"),
        ("governed by", "GOVERNS"),
        ("triggers", "TRIGGERED_BY"),
        ("triggered by", "TRIGGERED_BY"),
        ("publishes", "PUBLISHES_TO"),
        ("consumes", "CONSUMES_FROM"),
        ("routes", "ROUTES_TO"),
        ("stores", "STORES_IN"),
        ("persists", "STORES_IN"),
        ("monitors", "MONITORS"),
        ("validates", "VALIDATES"),
        ("uses", "USES"),
        ("receives", "RECEIVES_FROM"),
        ("processes", "PROCESSES"),
    )

    def __init__(self) -> None:
        """Initialize an empty NetworkX MultiDiGraph."""
        self.graph = nx.MultiDiGraph()
        self._entity_lookup = {
            name.casefold(): name for name in self.ENTITY_TYPES
        }

    @staticmethod
    def normalize_entity_name(name: str) -> str:
        """Normalize whitespace and casing-independent entity lookup."""
        return re.sub(r"\s+", " ", name.strip())

    def _canonical_entity_name(self, name: str) -> str:
        """Resolve an entity mention to its canonical name."""
        normalized = self.normalize_entity_name(name)
        return self._entity_lookup.get(normalized.casefold(), normalized)

    def add_entity(self, node: EntityNode) -> None:
        """Insert an entity or merge metadata into an existing node."""
        canonical_name = self._canonical_entity_name(node.name)
        entity_id = node.id.strip()

        if not entity_id:
            raise ValueError("Entity ID cannot be empty.")

        if entity_id in self.graph:
            existing = self.graph.nodes[entity_id]

            existing_metadata = dict(existing.get("metadata", {}))
            incoming_metadata = dict(node.metadata or {})

            document_ids = set(existing_metadata.get("document_ids", []))
            document_ids.update(incoming_metadata.get("document_ids", []))

            existing_metadata.update(incoming_metadata)

            if document_ids:
                existing_metadata["document_ids"] = sorted(document_ids)

            self.graph.nodes[entity_id].update(
                {
                    "name": canonical_name,
                    "entity_type": node.entity_type.value,
                    "description": node.description,
                    "metadata": existing_metadata,
                }
            )

            logger.debug("Updated existing entity: %s", entity_id)
            return

        self.graph.add_node(
            entity_id,
            name=canonical_name,
            entity_type=node.entity_type.value,
            description=node.description,
            metadata=dict(node.metadata or {}),
        )

        logger.debug("Added entity: %s", entity_id)

    def add_relation(self, edge: RelationEdge) -> None:
        """Insert a directed relationship while preventing duplicates."""
        source_id = edge.source_id.strip()
        target_id = edge.target_id.strip()
        relation_type = edge.relation_type.strip().upper()

        if source_id not in self.graph:
            raise ValueError(f"Source entity does not exist: {source_id}")

        if target_id not in self.graph:
            raise ValueError(f"Target entity does not exist: {target_id}")

        if source_id == target_id:
            raise ValueError("Self-referencing relationships are not allowed.")

        if not relation_type:
            raise ValueError("Relationship type cannot be empty.")

        existing_edges = self.graph.get_edge_data(source_id, target_id, default={})

        if relation_type in existing_edges:
            existing = existing_edges[relation_type]
            existing["weight"] = max(
                float(existing.get("weight", 1.0)),
                float(edge.weight),
            )
            logger.debug(
                "Updated existing relationship: %s -> %s (%s)",
                source_id,
                target_id,
                relation_type,
            )
            return

        self.graph.add_edge(
            source_id,
            target_id,
            key=relation_type,
            relation_type=relation_type,
            weight=float(edge.weight),
        )

        logger.debug(
            "Added relationship: %s -> %s (%s)",
            source_id,
            target_id,
            relation_type,
        )

    def _extract_entities(self, text: str) -> list[str]:
        """Find known entity names in a document sentence."""
        matches: list[tuple[int, str]] = []

        for entity_name in self.ENTITY_TYPES:
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_-]){re.escape(entity_name)}"
                rf"(?![A-Za-z0-9_-])",
                re.IGNORECASE,
            )

            for match in pattern.finditer(text):
                matches.append(
                    (match.start(), entity_name)
                )

        matches.sort(key=lambda item: item[0])

        result: list[str] = []
        seen: set[str] = set()

        for _, name in matches:
            canonical = self._canonical_entity_name(name)

            if canonical not in seen:
                result.append(canonical)
                seen.add(canonical)

        return result

    def _infer_relation_type(self, sentence: str) -> str:
        """Infer a relationship label using deterministic keyword rules."""
        normalized = sentence.casefold()

        for keyword, relation_type in self.RELATION_RULES:
            if keyword in normalized:
                return relation_type

        return "WORKS_WITH"

    def _process_document(self, document: dict[str, Any]) -> None:
        """Extract entities and relationships from one document."""
        document_id = document["id"]
        content = document["content"]

        sentences = re.split(r"(?<=[.!?])\s+", content.strip())

        for sentence in sentences:
            entities = self._extract_entities(sentence)

            for entity_name in entities:
                entity_type = self.ENTITY_TYPES[entity_name]

                entity = EntityNode(
                    id=entity_name,
                    name=entity_name,
                    entity_type=entity_type,
                    description=f"Entity referenced in document {document_id}.",
                    metadata={
                        "document_ids": [document_id],
                    },
                )

                self.add_entity(entity)

            if len(entities) < 2:
                continue

            relation_type = self._infer_relation_type(sentence)

            source_id = entities[0]

            for target_id in entities[1:]:
                if source_id == target_id:
                    continue

                relation = RelationEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    weight=1.0,
                )

                self.add_relation(relation)

    def build_from_corpus(
        self,
        documents_path: str,
    ) -> nx.MultiDiGraph:
        """Load JSON documents and construct the knowledge graph."""
        path = Path(documents_path)

        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as file:
                documents = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON corpus: {exc}") from exc

        if not isinstance(documents, list):
            raise TypeError("Corpus must contain a JSON list of documents.")

        for index, document in enumerate(documents):
            if not isinstance(document, dict):
                raise TypeError(f"Document at index {index} must be an object.")

            required_fields = {"id", "title", "category", "content"}

            missing = required_fields - document.keys()

            if missing:
                raise ValueError(
                    f"Document at index {index} is missing fields: "
                    f"{sorted(missing)}"
                )

            if not all(
                isinstance(document[field], str) and document[field].strip()
                for field in required_fields
            ):
                raise ValueError(
                    f"Document at index {index} contains empty or invalid fields."
                )

            self._process_document(document)

        logger.info(
            "Graph construction completed: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

        return self.graph

    def get_graph_stats(self) -> dict[str, Any]:
        """Return graph size, density, and connected component statistics."""
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()

        density = (
            nx.density(self.graph)
            if node_count > 1
            else 0.0
        )

        components = (
            nx.number_weakly_connected_components(self.graph)
            if node_count > 0
            else 0
        )

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "density": density,
            "connected_components": components,
        }


    def build_from_extractions(
        self,
        extractions: list[ExtractionResult],
    ) -> nx.MultiDiGraph:
        """Build or extend the graph using validated LLM extraction results.

        Entities appearing in multiple documents are merged, and their
        document IDs are retained as provenance metadata.
        """
        if not isinstance(extractions, list):
            raise TypeError("Extractions must be provided as a list.")

        # Validate the complete batch before modifying the graph.
        known_entities: dict[str, tuple[str, EntityType]] = {}

        for extraction in extractions:
            if not isinstance(extraction, ExtractionResult):
                raise TypeError(
                    "Every item must be an ExtractionResult instance."
                )

            entity_ids = [entity.id for entity in extraction.entities]

            if len(entity_ids) != len(set(entity_ids)):
                raise ValueError(
                    f"Duplicate entity IDs in extraction "
                    f"{extraction.document_id}."
                )

            current_entity_ids = set(entity_ids)

            for entity in extraction.entities:
                identity = (entity.name, entity.entity_type)
                document_ids = entity.metadata.get("document_ids", [])

                if not isinstance(document_ids, list) or not all(
                    isinstance(document_id, str)
                    for document_id in document_ids
                ):
                    raise ValueError(
                        f"Invalid document_ids metadata for entity: {entity.id}"
                    )

                if entity.id in known_entities:
                    if known_entities[entity.id] != identity:
                        raise ValueError(
                            f"Conflicting definitions for entity ID: "
                            f"{entity.id}"
                        )
                else:
                    known_entities[entity.id] = identity

                if entity.id in self.graph:
                    existing = self.graph.nodes[entity.id]
                    existing_identity = (
                        existing.get("name"),
                        EntityType(existing.get("entity_type")),
                    )
                    if existing_identity != identity:
                        raise ValueError(
                            f"Entity ID conflicts with an existing graph "
                            f"node: {entity.id}"
                        )

            for relation in extraction.relations:
                if relation.source_id not in current_entity_ids:
                    raise ValueError(
                        f"Relationship references an unknown source entity: "
                        f"{relation.source_id}"
                    )

                if relation.target_id not in current_entity_ids:
                    raise ValueError(
                        f"Relationship references an unknown target entity: "
                        f"{relation.target_id}"
                    )

                if relation.source_id == relation.target_id:
                    raise ValueError(
                        "Self-referencing relationships are not allowed."
                    )

        # Add entities and preserve the source document IDs.
        for extraction in extractions:
            for entity in extraction.entities:
                metadata = dict(entity.metadata)
                document_ids = set(metadata.get("document_ids", []))
                document_ids.add(extraction.document_id)
                metadata["document_ids"] = sorted(document_ids)

                self.add_entity(
                    entity.model_copy(update={"metadata": metadata})
                )

            for relation in extraction.relations:
                self.add_relation(relation)

        logger.info(
            "LLM graph construction completed: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

        return self.graph
