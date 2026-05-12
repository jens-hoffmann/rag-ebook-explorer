"""Shared fixtures for unit tests."""

import pytest
from langchain_core.documents import Document

from ebook_rag_explorer.services.chunking_service import ChunkingService


@pytest.fixture
def chunking_service() -> ChunkingService:
    """Create a chunking service with default settings."""
    return ChunkingService(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for testing."""
    return Document(
        page_content="This is a sample document with content for testing.",
        metadata={"page": 1, "source": "test.pdf"},
    )


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create a list of sample documents for testing."""
    return [
        Document(page_content="First document content " * 10, metadata={"doc": 1}),
        Document(page_content="Second document content " * 10, metadata={"doc": 2}),
        Document(page_content="Third document content " * 10, metadata={"doc": 3}),
    ]
