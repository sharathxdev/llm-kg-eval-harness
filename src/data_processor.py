"""
Document loading, cleaning, normalization, and validation.

The processor accepts JSON documents, validates required fields,
removes duplicate document IDs, and reports invalid records.
"""

import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class DocumentLoadError(Exception):
    """Raised when a document file cannot be loaded or parsed."""


class DocumentValidationError(ValueError):
    """Raised when a document does not satisfy the required schema."""


class DataProcessor:
    """
    Process raw documents into validated, normalized records.

    Required fields:
        id: Unique document identifier.
        title: Document title.
        category: Document category.
        content: Original document content.

    Invalid records are skipped by default and recorded in errors.
    Duplicate IDs are skipped; the first valid occurrence is retained.
    """

    REQUIRED_FIELDS = ("id", "title", "category", "content")

    def __init__(self, strict: bool = False) -> None:
        """
        Initialize the processor.

        Args:
            strict: If True, stop processing when an invalid record is found.
        """
        self.strict = strict
        self.errors: list[dict[str, Any]] = []
        self.duplicates: list[str] = []

    @staticmethod
    def normalize_whitespace(value: str) -> str:
        """
        Remove leading/trailing whitespace and collapse repeated whitespace.

        Args:
            value: Input string.

        Returns:
            String with normalized whitespace.
        """
        return " ".join(value.split())

    @classmethod
    def normalize_entity_name(cls, name: str) -> str:
        """
        Normalize an entity name for consistent comparison.

        The returned value is lowercase and has normalized whitespace.
        """
        return cls.normalize_whitespace(name).casefold()

    def validate_document(
        self,
        document: dict[str, Any],
        index: int = 0,
    ) -> dict[str, Any]:
        """
        Validate and normalize one document.

        Required text fields must be present, strings, and non-empty.
        The content field is preserved exactly as supplied.
        Optional metadata receives a safe empty-dictionary default.
        """
        if not isinstance(document, dict):
            raise DocumentValidationError(
                f"Record {index}: expected a JSON object, "
                f"received {type(document).__name__}."
            )

        missing_fields = [
            field
            for field in self.REQUIRED_FIELDS
            if field not in document
        ]

        if missing_fields:
            raise DocumentValidationError(
                f"Record {index}: missing required fields: "
                f"{', '.join(missing_fields)}."
            )

        cleaned = dict(document)

        for field in self.REQUIRED_FIELDS:
            value = cleaned[field]

            if not isinstance(value, str):
                raise DocumentValidationError(
                    f"Record {index}: field '{field}' must be a string."
                )

            if not value.strip():
                raise DocumentValidationError(
                    f"Record {index}: field '{field}' cannot be empty."
                )

        cleaned["id"] = self.normalize_whitespace(cleaned["id"])
        cleaned["title"] = self.normalize_whitespace(cleaned["title"])
        cleaned["category"] = self.normalize_whitespace(
            cleaned["category"]
        )

        # Preserve original content; only check that it is not whitespace.
        if not cleaned["content"].strip():
            raise DocumentValidationError(
                f"Record {index}: field 'content' cannot be empty."
            )

        # Optional metadata gets a safe default.
        if cleaned.get("metadata") is None:
            cleaned["metadata"] = {}
        elif "metadata" in cleaned and not isinstance(
            cleaned["metadata"], dict
        ):
            raise DocumentValidationError(
                f"Record {index}: field 'metadata' must be a JSON object."
            )

        return cleaned

    def process_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Validate, normalize, and deduplicate a list of documents.

        Invalid records are skipped unless strict mode is enabled.
        The first valid document for each ID is retained.
        """
        self.errors = []
        self.duplicates = []

        if not isinstance(documents, list):
            raise DocumentValidationError(
                "Expected the document collection to be a JSON list."
            )

        processed: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for index, document in enumerate(documents):
            try:
                cleaned = self.validate_document(document, index)

                document_id = cleaned["id"]

                if document_id in seen_ids:
                    self.duplicates.append(document_id)
                    logger.warning(
                        "Skipping duplicate document ID: %s",
                        document_id,
                    )
                    continue

                seen_ids.add(document_id)
                processed.append(cleaned)

            except DocumentValidationError as exc:
                error = {
                    "index": index,
                    "error": str(exc),
                }
                self.errors.append(error)

                logger.warning("Invalid document: %s", exc)

                if self.strict:
                    raise

        logger.info(
            "Document processing completed: %d valid, "
            "%d invalid, %d duplicates.",
            len(processed),
            len(self.errors),
            len(self.duplicates),
        )

        return processed

    def load_documents(
        self,
        file_path: str | Path,
    ) -> list[dict[str, Any]]:
        """
        Load, validate, normalize, and deduplicate documents from JSON.

        Supports either a top-level JSON list or an object containing
        a 'documents' list.
        """
        path = Path(file_path)

        try:
            with path.open("r", encoding="utf-8") as file:
                raw_data = json.load(file)

        except FileNotFoundError as exc:
            raise DocumentLoadError(
                f"Document file not found: {path}"
            ) from exc

        except json.JSONDecodeError as exc:
            raise DocumentLoadError(
                f"Malformed JSON in {path}: {exc}"
            ) from exc

        except OSError as exc:
            raise DocumentLoadError(
                f"Unable to read document file {path}: {exc}"
            ) from exc

        if isinstance(raw_data, dict):
            raw_data = raw_data.get("documents")

        if not isinstance(raw_data, list):
            raise DocumentLoadError(
                "JSON must contain a list of documents or an object "
                "with a 'documents' list."
            )

        return self.process_documents(raw_data)
