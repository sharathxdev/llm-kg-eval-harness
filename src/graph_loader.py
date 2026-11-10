"""Load and validate a knowledge graph exported as JSON."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
from pydantic import ValidationError

from src.graph_builder import KnowledgeGraphBuilder
from src.models import KnowledgeGraphSchema


def load_knowledge_graph(
    file_path: str | Path,
) -> nx.MultiDiGraph:
    """
    Load a validated knowledge graph from a JSON file.

    The JSON must contain nodes and edges compatible with
    KnowledgeGraphSchema. Additional top-level metadata is ignored.
    """

    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Knowledge graph JSON must contain a JSON object."
        )

    try:
        graph_schema = KnowledgeGraphSchema.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid knowledge graph data in {path}."
        ) from exc

    builder = KnowledgeGraphBuilder()

    for entity in graph_schema.nodes:
        builder.add_entity(entity)

    for relation in graph_schema.edges:
        builder.add_relation(relation)

    graph = builder.graph

    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("Loaded graph is not a NetworkX MultiDiGraph.")

    return graph
