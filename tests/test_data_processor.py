"""Unit tests for the DataProcessor."""

import json

import pytest

from src.data_processor import (
    DataProcessor,
    DocumentLoadError,
    DocumentValidationError,
)


@pytest.fixture
def processor():
    return DataProcessor()


@pytest.fixture
def valid_document():
    return {
        "id": "DOC-001",
        "title": "Core Cloud Architecture",
        "category": "Architecture",
        "content": "The API Gateway routes incoming requests.",
    }


def test_process_valid_document(processor, valid_document):
    result = processor.process_documents([valid_document])

    assert len(result) == 1
    assert result[0]["id"] == "DOC-001"
    assert result[0]["title"] == "Core Cloud Architecture"
    assert result[0]["metadata"] == {}


def test_normalize_whitespace(processor):
    result = processor.normalize_whitespace(
        "  Core   Cloud \n Architecture  "
    )

    assert result == "Core Cloud Architecture"


def test_normalize_entity_name(processor):
    result = processor.normalize_entity_name("  API   Gateway ")

    assert result == "api gateway"


def test_duplicate_document_ids_are_removed(
    processor,
    valid_document,
):
    duplicate = valid_document.copy()
    duplicate["title"] = "Duplicate Title"

    result = processor.process_documents(
        [valid_document, duplicate]
    )

    assert len(result) == 1
    assert result[0]["title"] == "Core Cloud Architecture"
    assert processor.duplicates == ["DOC-001"]


def test_missing_required_field_is_skipped(
    processor,
    valid_document,
):
    invalid = valid_document.copy()
    del invalid["title"]

    result = processor.process_documents(
        [valid_document, invalid]
    )

    assert len(result) == 1
    assert len(processor.errors) == 1
    assert "title" in processor.errors[0]["error"]


def test_empty_required_field_is_skipped(
    processor,
    valid_document,
):
    invalid = valid_document.copy()
    invalid["title"] = "   "

    result = processor.process_documents([invalid])

    assert result == []
    assert len(processor.errors) == 1


def test_non_string_required_field_is_rejected(
    processor,
    valid_document,
):
    invalid = valid_document.copy()
    invalid["id"] = 123

    result = processor.process_documents([invalid])

    assert result == []
    assert len(processor.errors) == 1


def test_invalid_json_raises_error(tmp_path):
    json_file = tmp_path / "invalid.json"
    json_file.write_text("{invalid json", encoding="utf-8")

    processor = DataProcessor()

    with pytest.raises(DocumentLoadError, match="Malformed JSON"):
        processor.load_documents(json_file)


def test_missing_file_raises_error(tmp_path):
    processor = DataProcessor()

    with pytest.raises(DocumentLoadError, match="not found"):
        processor.load_documents(tmp_path / "missing.json")


def test_load_documents_from_json(tmp_path, valid_document):
    json_file = tmp_path / "documents.json"
    json_file.write_text(
        json.dumps([valid_document]),
        encoding="utf-8",
    )

    processor = DataProcessor()
    result = processor.load_documents(json_file)

    assert len(result) == 1
    assert result[0]["id"] == "DOC-001"


def test_original_content_is_preserved(
    processor,
    valid_document,
):
    original_content = "  API Gateway\nroutes requests.  "
    valid_document["content"] = original_content

    result = processor.process_documents([valid_document])

    assert result[0]["content"] == original_content


def test_invalid_record_does_not_stop_batch(
    processor,
    valid_document,
):
    invalid = {"id": "DOC-002"}

    second_valid = valid_document.copy()
    second_valid["id"] = "DOC-003"

    result = processor.process_documents(
        [valid_document, invalid, second_valid]
    )

    assert len(result) == 2
    assert len(processor.errors) == 1


def test_strict_mode_raises_validation_error(valid_document):
    processor = DataProcessor(strict=True)
    invalid = {"id": "DOC-002"}

    with pytest.raises(DocumentValidationError):
        processor.process_documents([valid_document, invalid])


def test_non_list_input_raises_validation_error(processor):
    with pytest.raises(DocumentValidationError):
        processor.process_documents({"id": "DOC-001"})


def test_document_wrapper_format_is_supported(
    tmp_path,
    valid_document,
):
    json_file = tmp_path / "wrapped.json"
    json_file.write_text(
        json.dumps({"documents": [valid_document]}),
        encoding="utf-8",
    )

    processor = DataProcessor()
    result = processor.load_documents(json_file)

    assert len(result) == 1
    assert result[0]["id"] == "DOC-001"
