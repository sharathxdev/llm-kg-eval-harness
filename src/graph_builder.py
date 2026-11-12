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

    RELATION_RULES: ClassVar[tuple[tuple[str, str], ...]] = (
        ("communicating with", "WORKS_WITH"),
        ("communicates with", "WORKS_WITH"),
        ("coordinates with", "WORKS_WITH"),
        ("works with", "WORKS_WITH"),
        ("depends on", "DEPENDS_ON"),
        ("depend on", "DEPENDS_ON"),
        ("affecting", "AFFECTS"),
        ("affects", "AFFECTS"),
        ("impacts", "AFFECTS"),
        ("governed by", "GOVERNS"),
        ("governs", "GOVERNS"),
        ("triggered by", "TRIGGERED_BY"),
        ("triggers", "TRIGGERED_BY"),
        ("publishes", "PUBLISHES_TO"),
        ("consumes", "CONSUMES_FROM"),
        ("routes", "ROUTES_TO"),
        ("stores", "STORES_IN"),
        ("caches", "CACHES_IN"),
        ("publish", "PUBLISHES_TO"),
        ("persists", "STORES_IN"),
        ("monitors", "MONITORS"),
        ("validates", "VALIDATES"),
        ("uses", "USES"),
        ("receives", "RECEIVES_FROM"),
        ("processes", "PROCESSES"),
        ("coordinates", "WORKS_WITH"),
    )

    # These predicates require a particular preposition before the
    # target entity. This prevents unrelated entities later in a
    # sentence from being treated as relationship targets.
    RELATION_CONNECTORS: ClassVar[dict[str, tuple[str, ...]]] = {
        "publishes": ("to",),
        "consumes": ("from",),
        "routes": ("to",),
        "stores": ("in",),
        "caches": ("in",),
        "publish": ("to",),
        "persists": ("in",),
        "receives": ("from", "through"),
        "coordinates": ("with",),
    }

    # These phrases can directly precede their target entity.
    DIRECT_OBJECT_RELATIONS: ClassVar[set[str]] = {
        "communicating with",
        "communicates with",
        "coordinates with",
        "works with",
        "depends on",
        "depend on",
        "affecting",
        "affects",
        "impacts",
        "governed by",
        "governs",
        "triggered by",
        "triggers",
        "monitors",
        "validates",
        "uses",
        "processes",
        "coordinates",
    }


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

    def add_relation(
        self,
        edge: RelationEdge,
        *,
        document_id: str | None = None,
        source_sentence: str | None = None,
    ) -> None:
        """Insert a directed relationship and merge its provenance."""
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

        existing_edges = self.graph.get_edge_data(
            source_id,
            target_id,
            default={},
        )

        provenance_values = {}

        if document_id:
            provenance_values["document_ids"] = [document_id]

        if source_sentence:
            provenance_values["source_sentences"] = [source_sentence.strip()]

        if relation_type in existing_edges:
            existing = existing_edges[relation_type]

            existing["weight"] = max(
                float(existing.get("weight", 1.0)),
                float(edge.weight),
            )

            if provenance_values:
                provenance = dict(existing.get("provenance", {}))

                for field, incoming_values in provenance_values.items():
                    values = set(provenance.get(field, []))
                    values.update(incoming_values)
                    provenance[field] = sorted(values)

                existing["provenance"] = provenance

            logger.debug(
                "Updated existing relationship: %s -> %s (%s)",
                source_id,
                target_id,
                relation_type,
            )
            return

        attributes = {
            "relation_type": relation_type,
            "weight": float(edge.weight),
        }

        if provenance_values:
            attributes["provenance"] = provenance_values

        self.graph.add_edge(
            source_id,
            target_id,
            key=relation_type,
            **attributes,
        )

        logger.debug(
            "Added relationship: %s -> %s (%s)",
            source_id,
            target_id,
            relation_type,
        )


    def _extract_entity_mentions(
        self,
        text: str,
    ) -> list[tuple[int, int, str]]:
        """Return every known entity occurrence with its text positions."""
        matches: list[tuple[int, int, str]] = []

        for entity_name in self.ENTITY_TYPES:
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_-]){re.escape(entity_name)}"
                rf"(?![A-Za-z0-9_-])",
                re.IGNORECASE,
            )

            for match in pattern.finditer(text):
                canonical = self._canonical_entity_name(entity_name)
                matches.append((match.start(), match.end(), canonical))

        matches.sort(key=lambda item: (item[0], item[1]))

        return matches

    def _extract_entities(self, text: str) -> list[str]:
        """Find unique known entity names in a sentence."""
        result: list[str] = []
        seen: set[str] = set()

        for _, _, name in self._extract_entity_mentions(text):
            if name not in seen:
                result.append(name)
                seen.add(name)

        return result

    def _infer_relation_type(self, sentence: str) -> str | None:
        """Infer a relationship label when a supported phrase is present."""
        normalized = sentence.casefold()

        for keyword, relation_type in self.RELATION_RULES:
            if keyword in normalized:
                return relation_type

        # Entity co-occurrence alone is not evidence of a relationship.
        return None

    def _extract_relationships(
        self,
        sentence: str,
        *,
        subject_hint: str | None = None,
    ) -> list[tuple[str, str, str]]:
        """Extract relationships only when their endpoints are supported.

        The subject is the entity associated with the relationship phrase.
        The target must follow the expected connector or directly follow a
        supported direct-object relationship phrase.

        Coordinated targets are supported, but unrelated entities elsewhere
        in the sentence are not automatically connected.
        """
        mentions = self._extract_entity_mentions(sentence)

        if not mentions or (len(mentions) < 2 and not subject_hint):
            return []

        # Find relationship phrases, preferring longer phrases when rules
        # overlap, such as "coordinates with" and "coordinates".
        candidates = []

        for keyword, relation_type in self.RELATION_RULES:
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_-]){re.escape(keyword)}"
                rf"(?![A-Za-z0-9_-])",
                re.IGNORECASE,
            )

            for match in pattern.finditer(sentence):
                candidates.append(
                    (
                        match.start(),
                        match.end(),
                        keyword,
                        relation_type,
                        match,
                    )
                )

        candidates.sort(
            key=lambda item: (item[0], -(item[1] - item[0]))
        )

        relation_matches = []

        for candidate in candidates:
            start, end = candidate[0], candidate[1]

            overlaps = any(
                start < existing[1] and end > existing[0]
                for existing in relation_matches
            )

            if not overlaps:
                relation_matches.append(candidate)

        relation_matches.sort(key=lambda item: item[0])

        extracted: list[tuple[str, str, str]] = []
        previous_relation_end = None
        previous_subject = None

        for index, candidate in enumerate(relation_matches):
            start, end, keyword, relation_type, _ = candidate

            preceding_mentions = [
                mention
                for mention in mentions
                if mention[1] <= start
            ]

            nearest_subject = (
                preceding_mentions[-1]
                if preceding_mentions
                else None
            )

            # Resolve a relative clause to the nearest entity before
            # "which", rather than retaining an earlier coordinated subject.
            latest_which = sentence.casefold().rfind("which", 0, start)
            last_boundary = max(
                sentence.rfind(".", 0, start),
                sentence.rfind(";", 0, start),
            )
            is_relative_clause = latest_which > last_boundary

            if is_relative_clause:
                antecedents = [
                    mention
                    for mention in mentions
                    if mention[1] <= latest_which
                ]

                if not antecedents:
                    previous_relation_end = end
                    continue

                subject_id = antecedents[-1][2]

            elif nearest_subject is None:
                # Use the preceding sentence's subject only if this
                # sentence explicitly begins with "It".
                begins_with_it = bool(
                    re.match(r"^\s*It\b", sentence, re.IGNORECASE)
                )

                if subject_hint and begins_with_it:
                    subject_id = subject_hint
                else:
                    previous_relation_end = end
                    previous_subject = None
                    continue

            else:
                subject_id = nearest_subject[2]

                # Preserve the subject for coordinated predicates:
                # "X depends on A and publishes events to B".
                if (
                    previous_subject is not None
                    and previous_relation_end is not None
                ):
                    bridge = sentence[previous_relation_end:start]

                    has_conjunction = bool(
                        re.search(r"\band\b", bridge, re.IGNORECASE)
                    )
                    has_clause_boundary = bool(
                        re.search(
                            r"[.!?;]|\b(?:while|but|whereas|although)\b",
                            bridge,
                            re.IGNORECASE,
                        )
                    )

                    if has_conjunction and not has_clause_boundary:
                        subject_id = previous_subject

            previous_subject = subject_id
            previous_relation_end = end

            next_relation_start = (
                relation_matches[index + 1][0]
                if index + 1 < len(relation_matches)
                else len(sentence)
            )

            target_candidates = [
                mention
                for mention in mentions
                if end <= mention[0] < next_relation_start
            ]

            if not target_candidates:
                continue

            first_target = None
            connectors = self.RELATION_CONNECTORS.get(keyword, ())

            for target in target_candidates:
                gap = sentence[end:target[0]]

                has_connector = any(
                    re.search(
                        rf"\b{re.escape(connector)}\b\s*$",
                        gap,
                        re.IGNORECASE,
                    )
                    for connector in connectors
                )

                normalized_gap = gap.strip().casefold()

                has_direct_object = (
                    keyword in self.DIRECT_OBJECT_RELATIONS
                    and normalized_gap in {"", "the", "a", "an"}
                )

                # "coordinates customer orders by communicating with X"
                # must not be mistaken for "coordinates with X".
                if keyword == "coordinates" and re.search(
                    r"\bby\b",
                    gap,
                    re.IGNORECASE,
                ):
                    has_connector = False
                    has_direct_object = False

                if has_connector or has_direct_object:
                    first_target = target
                    break

            if first_target is None:
                continue

            extracted.append(
                (subject_id, first_target[2], relation_type)
            )

            previous_target = first_target

            # Support coordinated targets, for example:
            # "depends on RedisCache and PostgresCluster".
            for target in target_candidates:
                if target[0] <= first_target[0]:
                    continue

                between_targets = sentence[
                    previous_target[1]:target[0]
                ]

                has_coordination = bool(
                    re.search(r"\band\b|\bor\b|,", between_targets)
                )

                has_clause_boundary = bool(
                    re.search(
                        r"[.!?;]|\b(?:which|while|but|whereas)\b",
                        between_targets,
                        re.IGNORECASE,
                    )
                )

                if not has_coordination or has_clause_boundary:
                    break

                extracted.append(
                    (subject_id, target[2], relation_type)
                )
                previous_target = target

        # Preserve order while removing duplicate triples.
        return list(dict.fromkeys(extracted))

    def _process_document(self, document: dict[str, Any]) -> None:
        """Extract entities and supported relationships from one document."""
        document_id = document["id"]
        content = document["content"]

        sentences = re.split(r"(?<=[.!?])\s+", content.strip())
        previous_subject: str | None = None

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            begins_with_it = bool(
                re.match(r"^\s*It\b", sentence, re.IGNORECASE)
            )

            subject_hint = previous_subject if begins_with_it else None

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

            for source_id, target_id, relation_type in (
                self._extract_relationships(
                    sentence,
                    subject_hint=subject_hint,
                )
            ):
                relation = RelationEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    weight=1.0,
                )

                self.add_relation(
                    relation,
                    document_id=document_id,
                    source_sentence=sentence,
                )

            # Update the remembered subject only when an entity clearly
            # begins the sentence (optionally after "the", "a", or "an").
            # Do not mistake the first object in an "It ..." sentence
            # for that sentence's subject.
            if not begins_with_it:
                mentions = self._extract_entity_mentions(sentence)

                if mentions:
                    first_mention = mentions[0]
                    prefix = sentence[:first_mention[0]]

                    if re.fullmatch(
                        r"\s*(?:(?:the|a|an)\s+)?",
                        prefix,
                        re.IGNORECASE,
                    ):
                        previous_subject = first_mention[2]


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
                self.add_relation(
                    relation,
                    document_id=extraction.document_id,
                )

        logger.info(
            "LLM graph construction completed: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

        return self.graph
