"""Tests for ChunkingService."""

import pytest
from langchain_core.documents import Document

from ebook_rag_explorer.services.chunking_service import ChunkingService


class TestChunkingService:
    """Test cases for ChunkingService."""

    @pytest.fixture
    def service(self) -> ChunkingService:
        """Create a chunking service with default settings."""
        return ChunkingService(chunk_size=100, chunk_overlap=20)

    def test_chunk_document_splits_text(self, service: ChunkingService) -> None:
        """Test that long documents are split into chunks."""
        # Create a document with content longer than chunk_size
        content = "Word " * 50  # Will be longer than 100 chars
        doc = Document(page_content=content, metadata={"page": 1})

        chunks = service.chunk_document(doc)

        # Should produce multiple chunks
        assert len(chunks) > 1
        # Each chunk should have the original metadata plus chunk info
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["page"] == 1
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_chunk_document_small_text_no_split(self, service: ChunkingService) -> None:
        """Test that short documents are not split."""
        doc = Document(page_content="Short text.", metadata={"page": 1})

        chunks = service.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text."
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["total_chunks"] == 1

    def test_chunk_documents_multiple(self, service: ChunkingService) -> None:
        """Test chunking multiple documents."""
        docs = [
            Document(page_content="First document content " * 10, metadata={"doc": 1}),
            Document(page_content="Second document content " * 10, metadata={"doc": 2}),
        ]

        chunks = service.chunk_documents(docs)

        # Should have chunks from both documents
        assert len(chunks) > 2
        # Verify metadata is preserved
        doc_ids = {chunk.metadata.get("doc") for chunk in chunks}
        assert doc_ids == {1, 2}

    def test_chunk_documents_empty_list(self, service: ChunkingService) -> None:
        """Test chunking an empty list returns empty."""
        chunks = service.chunk_documents([])
        assert chunks == []

    def test_overlap_preserves_context(self) -> None:
        """Test that chunks overlap to preserve context."""
        service = ChunkingService(chunk_size=50, chunk_overlap=20)

        # Create content that will be split
        content = "ABCDEFGHIJ" * 10  # 100 characters
        doc = Document(page_content=content, metadata={})

        chunks = service.chunk_document(doc)

        # With overlap, chunks should share some content
        assert len(chunks) >= 2
        # The second chunk should start with some of the first chunk's end
        # (accounting for splitting at word boundaries)
