"""Tests for LLM-based structured extraction."""

import json

import pytest

from src.extractor import LLMEntityExtractor, LLMExtractionError

DOCUMENT = {
    "id": "DOC-TEST-001",
    "title": "Payment Architecture",
    "category": "Architecture",
    "content": (
        "PaymentGateway depends on AuthService "
        "for authorization."
    ),
}


VALID_EXTRACTION = {
    "document_id": "DOC-TEST-001",
    "entities": [
        {
            "id": "PaymentGateway",
            "name": "PaymentGateway",
            "entity_type": "SERVICE",
            "description": "Processes payments.",
            "metadata": {"document_ids": ["DOC-TEST-001"]},
        },
        {
            "id": "AuthService",
            "name": "AuthService",
            "entity_type": "SERVICE",
            "description": "Provides authentication.",
            "metadata": {"document_ids": ["DOC-TEST-001"]},
        },
    ],
    "relations": [
        {
            "source_id": "PaymentGateway",
            "target_id": "AuthService",
            "relation_type": "depends_on",
            "weight": 1.0,
        }
    ],
}


class MockMessage:
    def __init__(self, content):
        self.content = content


class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)


class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]


class MockCompletions:
    def __init__(self, content):
        self.content = content
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return MockResponse(self.content)


class MockChat:
    def __init__(self, content):
        self.completions = MockCompletions(content)


class MockClient:
    def __init__(self, content):
        self.chat = MockChat(content)


@pytest.fixture
def extractor():
    client = MockClient(json.dumps(VALID_EXTRACTION))
    return LLMEntityExtractor(client=client)


def test_successful_extraction(extractor):
    result = extractor.extract(DOCUMENT)

    assert result.document_id == "DOC-TEST-001"
    assert len(result.entities) == 2
    assert len(result.relations) == 1
    assert result.relations[0].relation_type == "DEPENDS_ON"


def test_extraction_returns_pydantic_model(extractor):
    result = extractor.extract(DOCUMENT)

    assert result.__class__.__name__ == "ExtractionResult"


def test_request_uses_json_response_format(extractor):
    extractor.extract(DOCUMENT)

    request = extractor.client.chat.completions.last_request

    assert request["response_format"] == {"type": "json_object"}
    assert request["model"] == "gpt-4o-mini"


def test_prompt_contains_document_content(extractor):
    extractor.extract(DOCUMENT)

    request = extractor.client.chat.completions.last_request
    prompt = request["messages"][1]["content"]

    assert "PaymentGateway" in prompt
    assert "AuthService" in prompt


def test_empty_document_id_is_rejected(extractor):
    document = {**DOCUMENT, "id": "   "}

    with pytest.raises(ValueError, match="non-empty string ID"):
        extractor.extract(document)


def test_empty_document_content_is_rejected(extractor):
    document = {**DOCUMENT, "content": "   "}

    with pytest.raises(ValueError, match="non-empty string content"):
        extractor.extract(document)


def test_non_dictionary_document_is_rejected(extractor):
    with pytest.raises(TypeError, match="must be a dictionary"):
        extractor.extract("invalid")


def test_invalid_json_response_raises_error():
    extractor = LLMEntityExtractor(
        client=MockClient("not valid JSON")
    )

    with pytest.raises(LLMExtractionError, match="invalid JSON"):
        extractor.extract(DOCUMENT)


def test_empty_llm_response_raises_error():
    extractor = LLMEntityExtractor(client=MockClient(""))

    with pytest.raises(LLMExtractionError, match="empty extraction response"):
        extractor.extract(DOCUMENT)


def test_mismatched_document_id_raises_error():
    payload = {**VALID_EXTRACTION, "document_id": "DOC-WRONG"}
    extractor = LLMEntityExtractor(
        client=MockClient(json.dumps(payload))
    )

    with pytest.raises(LLMExtractionError, match="does not match"):
        extractor.extract(DOCUMENT)


def test_invalid_entity_type_raises_error():
    payload = json.loads(json.dumps(VALID_EXTRACTION))
    payload["entities"][0]["entity_type"] = "UNKNOWN"

    extractor = LLMEntityExtractor(
        client=MockClient(json.dumps(payload))
    )

    with pytest.raises(LLMExtractionError, match="schema validation"):
        extractor.extract(DOCUMENT)


def test_unknown_relationship_source_raises_error():
    payload = json.loads(json.dumps(VALID_EXTRACTION))
    payload["relations"][0]["source_id"] = "UnknownService"

    extractor = LLMEntityExtractor(
        client=MockClient(json.dumps(payload))
    )

    with pytest.raises(LLMExtractionError, match="unknown source entity"):
        extractor.extract(DOCUMENT)


def test_unknown_relationship_target_raises_error():
    payload = json.loads(json.dumps(VALID_EXTRACTION))
    payload["relations"][0]["target_id"] = "UnknownDatabase"

    extractor = LLMEntityExtractor(
        client=MockClient(json.dumps(payload))
    )

    with pytest.raises(LLMExtractionError, match="unknown target entity"):
        extractor.extract(DOCUMENT)


def test_self_referencing_relationship_is_rejected():
    payload = json.loads(json.dumps(VALID_EXTRACTION))
    payload["relations"][0]["target_id"] = "PaymentGateway"

    extractor = LLMEntityExtractor(
        client=MockClient(json.dumps(payload))
    )

    with pytest.raises(LLMExtractionError, match="Self-referencing"):
        extractor.extract(DOCUMENT)


def test_missing_api_key_raises_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    extractor = LLMEntityExtractor(client=None)

    with pytest.raises(LLMExtractionError, match="OPENAI_API_KEY"):
        extractor.extract(DOCUMENT)


def test_llm_api_failure_is_wrapped():
    class FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("Simulated API failure")

    class FailingChat:
        completions = FailingCompletions()

    class FailingClient:
        chat = FailingChat()

    extractor = LLMEntityExtractor(client=FailingClient())

    with pytest.raises(LLMExtractionError, match="LLM extraction failed"):
        extractor.extract(DOCUMENT)


def test_system_prompt_defines_only_supported_relationship_types():
    prompt = LLMEntityExtractor.SYSTEM_PROMPT

    supported_types = [
        "DEPENDS_ON",
        "AFFECTS",
        "TRIGGERED_BY",
        "GOVERNS",
        "MANAGES",
        "MONITORS",
        "OPERATES",
    ]

    for relation_type in supported_types:
        assert relation_type in prompt

    assert "PUBLISHES_TO" not in prompt
    assert "CONSUMES_FROM" not in prompt


def test_system_prompt_defines_incident_relationship_direction():
    prompt = LLMEntityExtractor.SYSTEM_PROMPT

    assert "IncidentID -> AFFECTS -> AffectedEntityID" in prompt
    assert "IncidentID -> TRIGGERED_BY -> CauseEntityID" in prompt
    assert "Do not reverse an incident relationship" in prompt


def test_system_prompt_requires_exact_entity_name_preservation():
    prompt = LLMEntityExtractor.SYSTEM_PROMPT

    assert "exact spelling, capitalization, and punctuation" in prompt
    assert "including capitalization" in prompt


def test_system_prompt_rejects_unsupported_relationship_inference():
    prompt = LLMEntityExtractor.SYSTEM_PROMPT

    assert "Do not invent entities, entity aliases, or relationships" in prompt
    assert "Do not add relationships merely because they seem technically plausible" in prompt
    assert "return an empty relations list" in prompt
