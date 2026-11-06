
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EntityType(str, Enum):
    """Supported entity categories in the knowledge graph."""

    SERVICE = "SERVICE"
    DATABASE = "DATABASE"
    POLICY = "POLICY"
    INCIDENT = "INCIDENT"
    TEAM = "TEAM"


class EntityNode(BaseModel):
    """Represents an entity stored as a knowledge graph node."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(..., min_length=1, description="Unique entity identifier")
    name: str = Field(..., min_length=1, description="Human-readable entity name")
    entity_type: EntityType = Field(..., description="Entity category")
    description: str = Field(
        default="",
        description="Description of the entity"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional entity attributes and provenance"
    )


class RelationEdge(BaseModel):
    """Represents a directed relationship between two entities."""

    model_config = ConfigDict(str_strip_whitespace=True)

    source_id: str = Field(..., min_length=1, description="Source entity ID")
    target_id: str = Field(..., min_length=1, description="Target entity ID")
    relation_type: str = Field(
        ...,
        min_length=1,
        description="Relationship label such as DEPENDS_ON or AFFECTS"
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Relationship strength"
    )

    @field_validator("relation_type")
    @classmethod
    def normalize_relation_type(cls, value: str) -> str:
        return value.strip().upper()


class KnowledgeGraphSchema(BaseModel):
    """Complete knowledge graph containing nodes and relationships."""

    nodes: list[EntityNode] = Field(
        default_factory=list,
        description="Knowledge graph entities"
    )
    edges: list[RelationEdge] = Field(
        default_factory=list,
        description="Directed entity relationships"
    )


class ExtractionResult(BaseModel):
    """Structured extraction output from a source document."""

    document_id: str = Field(..., min_length=1)
    entities: list[EntityNode] = Field(default_factory=list)
    relations: list[RelationEdge] = Field(default_factory=list)


class EvalSample(BaseModel):
    """Evaluation sample containing expected extraction results."""

    id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    ground_truth: str = Field(..., min_length=1)
    expected_entities: list[str] = Field(default_factory=list)
    expected_relations: list[str] = Field(default_factory=list)

