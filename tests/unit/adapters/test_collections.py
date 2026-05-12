"""Tests for collections functionality."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from ebook_rag_explorer.adapters.vectorstore.chroma_adapter import (
    ChromaAdapter,
    normalize_collection_id,
)


class TestNormalizeCollectionId:
    """Tests for collection ID normalization."""

    def test_normalize_lowercase(self):
        """Test that normalization converts to lowercase."""
        assert normalize_collection_id("MyCollection") == "mycollection"

    def test_normalize_strips_whitespace(self):
        """Test that normalization strips whitespace."""
        assert normalize_collection_id("  my collection  ") == "my_collection"

    def test_normalize_replaces_special_chars(self):
        """Test that special characters are replaced with underscores."""
        assert normalize_collection_id("my@collection#123") == "my_collection_123"

    def test_normalize_consecutive_underscores(self):
        """Test that consecutive underscores are collapsed."""
        assert normalize_collection_id("my___collection") == "my_collection"

    def test_normalize_strips_leading_trailing_underscores(self):
        """Test that leading/trailing underscores are stripped."""
        assert normalize_collection_id("_my_collection_") == "my_collection"

    def test_normalize_empty_fallback(self):
        """Test that empty string falls back to 'default'."""
        assert normalize_collection_id("") == "default"
        assert normalize_collection_id("   ") == "default"
        assert normalize_collection_id("___") == "default"


class TestChromaAdapterCollections:
    """Tests for ChromaDB adapter collection functionality."""

    @pytest.fixture
    def temp_persist_dir(self, tmp_path):
        """Create a temporary directory for ChromaDB."""
        return tmp_path

    @pytest.fixture
    def adapter(self, temp_persist_dir):
        """Create a ChromaAdapter instance."""
        return ChromaAdapter(temp_persist_dir)

    @pytest.fixture
    def sample_docs(self):
        """Create sample documents."""
        return [
            Document(page_content="Test content 1", metadata={}),
            Document(page_content="Test content 2", metadata={}),
        ]

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    def test_add_documents_with_collection(self, adapter, sample_docs, sample_embeddings):
        """Test adding documents with collection_id."""
        adapter.add_documents(
            sample_docs,
            sample_embeddings,
            book_id="book1",
            collection_id="My Collection",
        )

        # Verify collection was stored (normalized)
        books = adapter.list_books()
        assert len(books) == 1
        assert books[0]["collection_id"] == "my_collection"

    def test_list_collections_empty(self, adapter):
        """Test listing collections when none exist."""
        collections = adapter.list_collections()
        assert collections == []

    def test_list_collections_with_data(self, adapter, sample_docs, sample_embeddings):
        """Test listing collections with data."""
        # Add documents to different collections
        adapter.add_documents(
            sample_docs,
            sample_embeddings,
            book_id="book1",
            collection_id="Collection A",
        )
        adapter.add_documents(
            [Document(page_content="More content", metadata={})],
            [[0.0, 0.0, 1.0]],
            book_id="book2",
            collection_id="Collection A",  # Same collection
        )
        adapter.add_documents(
            sample_docs,
            sample_embeddings,
            book_id="book3",
            collection_id="Collection B",  # Different collection
        )

        collections = adapter.list_collections()
        assert len(collections) == 2

        # Find Collection A
        coll_a = next(c for c in collections if c["id"] == "collection_a")
        assert coll_a["book_count"] == 2  # book1 and book2
        assert coll_a["chunk_count"] == 3  # 2 + 1 chunks

        # Find Collection B
        coll_b = next(c for c in collections if c["id"] == "collection_b")
        assert coll_b["book_count"] == 1  # book3
        assert coll_b["chunk_count"] == 2

    def test_delete_collection_success(self, adapter, sample_docs, sample_embeddings):
        """Test successfully deleting a collection."""
        # Add documents to collection
        adapter.add_documents(
            sample_docs,
            sample_embeddings,
            book_id="book1",
            collection_id="test-collection",
        )

        # Verify it exists
        assert len(adapter.list_collections()) == 1

        # Delete the collection
        result = adapter.delete_collection("test-collection")
        assert result is True

        # Verify it's gone
        assert len(adapter.list_collections()) == 0

    def test_delete_collection_not_found(self, adapter):
        """Test deleting a non-existent collection."""
        result = adapter.delete_collection("non-existent")
        assert result is False

    def test_delete_collection_normalizes_id(self, adapter, sample_docs, sample_embeddings):
        """Test that collection_id is normalized when deleting."""
        adapter.add_documents(
            sample_docs,
            sample_embeddings,
            book_id="book1",
            collection_id="Test Collection",
        )

        # Delete with different casing/spacing
        result = adapter.delete_collection("test_collection")
        assert result is True

    def test_similarity_search_with_collection_filter(
        self, adapter, sample_docs, sample_embeddings
    ):
        """Test searching within a specific collection."""
        # Add to collection A
        adapter.add_documents(
            [Document(page_content="Python programming", metadata={})],
            [[1.0, 0.0, 0.0]],
            book_id="book1",
            collection_id="Collection A",
        )

        # Add to collection B
        adapter.add_documents(
            [Document(page_content="Machine learning", metadata={})],
            [[0.0, 1.0, 0.0]],
            book_id="book2",
            collection_id="Collection B",
        )

        # Search in Collection A
        results = adapter.similarity_search(
            query_embedding=[1.0, 0.0, 0.0],
            k=10,
            collection_id="Collection A",
        )

        assert len(results) == 1
        assert "Python" in results[0].page_content

    def test_similarity_search_with_book_and_collection_filter(self, adapter):
        """Test searching with both book_id and collection_id filters."""
        # Add multiple books to same collection
        adapter.add_documents(
            [Document(page_content="Book 1 content", metadata={})],
            [[1.0, 0.0, 0.0]],
            book_id="book1",
            collection_id="MyCollection",
        )
        adapter.add_documents(
            [Document(page_content="Book 2 content", metadata={})],
            [[0.5, 0.5, 0.0]],
            book_id="book2",
            collection_id="MyCollection",
        )

        # Search specific book in collection
        results = adapter.similarity_search(
            query_embedding=[1.0, 0.0, 0.0],
            k=10,
            book_id="book1",
            collection_id="MyCollection",
        )

        assert len(results) == 1
        assert "Book 1" in results[0].page_content
