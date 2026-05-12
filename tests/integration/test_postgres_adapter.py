"""Integration tests for PostgreSQL adapter using testcontainers."""

import pytest
from langchain_core.documents import Document

from ebook_rag_explorer.adapters.vectorstore.postgres_adapter import normalize_collection_id


class TestNormalizeCollectionId:
    """Tests for collection ID normalization."""

    def test_normalize_lowercase(self):
        assert normalize_collection_id("MyCollection") == "mycollection"

    def test_normalize_strips_whitespace(self):
        assert normalize_collection_id("  my collection  ") == "my_collection"

    def test_normalize_replaces_special_chars(self):
        assert normalize_collection_id("my@collection#123") == "my_collection_123"

    def test_normalize_empty_fallback(self):
        assert normalize_collection_id("") == "default"
        assert normalize_collection_id("   ") == "default"


class TestPostgresAdapterBasic:
    """Basic tests for PostgreSQL adapter."""

    async def test_add_documents(self, adapter):
        """Test adding documents to the database."""
        docs = [
            Document(page_content="Python is a great programming language.", metadata={"page": 1}),
            Document(page_content="Machine learning uses Python extensively.", metadata={"page": 2}),
        ]
        embeddings = [
            [0.1, 0.2, 0.3] + [0.0] * 381,
            [0.4, 0.5, 0.6] + [0.0] * 381,
        ]

        await adapter.add_documents(docs, embeddings, book_id="book1")

        books = await adapter.list_books()
        assert len(books) == 1
        assert books[0]["id"] == "book1"
        assert books[0]["chunk_count"] == 2

    async def test_add_documents_with_collection(self, adapter):
        """Test adding documents with a collection."""
        docs = [Document(page_content="Test content", metadata={})]
        embeddings = [[0.1] * 384]

        await adapter.add_documents(docs, embeddings, book_id="book2", collection_id="My Collection")

        books = await adapter.list_books()
        assert books[0]["collection_id"] == "my_collection"

    async def test_delete_book(self, adapter):
        """Test deleting a book."""
        docs = [Document(page_content="Content to delete", metadata={})]
        embeddings = [[0.1] * 384]

        await adapter.add_documents(docs, embeddings, book_id="book3")

        books = await adapter.list_books()
        assert len(books) == 3

        result = await adapter.delete_book("book3")
        assert result is True

        books = await adapter.list_books()
        assert len(books) == 2

    async def test_delete_book_not_found(self, adapter):
        """Test deleting a non-existent book."""
        result = await adapter.delete_book("nonexistent")
        assert result is False

    async def test_list_books_empty(self, adapter):
        """Test listing books when none exist."""
        await adapter.clear()
        books = await adapter.list_books()
        assert books == []

    async def test_clear(self, adapter):
        """Test clearing all documents."""
        docs = [Document(page_content="Content", metadata={})]
        embeddings = [[0.1] * 384]
        await adapter.add_documents(docs, embeddings, book_id="cleartest")

        await adapter.clear()

        books = await adapter.list_books()
        assert books == []


class TestPostgresAdapterCollections:
    """Tests for collection functionality."""

    async def test_list_collections_empty(self, adapter):
        """Test listing collections when none exist."""
        await adapter.clear()
        collections = await adapter.list_collections()
        assert collections == []

    async def test_list_collections_with_data(self, adapter):
        """Test listing collections with data."""
        docs1 = [Document(page_content="Python content", metadata={})]
        emb1 = [[0.1] * 384]
        await adapter.add_documents(docs1, emb1, book_id="pybook1", collection_id="Programming")

        docs2 = [Document(page_content="More Python content", metadata={})]
        emb2 = [[0.2] * 384]
        await adapter.add_documents(docs2, emb2, book_id="pybook2", collection_id="Programming")

        docs3 = [Document(page_content="Cooking content", metadata={})]
        emb3 = [[0.3] * 384]
        await adapter.add_documents(docs3, emb3, book_id="cookbook1", collection_id="Cooking")

        collections = await adapter.list_collections()

        assert len(collections) == 2

        prog = next(c for c in collections if c["id"] == "programming")
        assert prog["book_count"] == 2
        assert prog["chunk_count"] == 2

        cooking = next(c for c in collections if c["id"] == "cooking")
        assert cooking["book_count"] == 1
        assert cooking["chunk_count"] == 1

        await adapter.clear()

    async def test_delete_collection(self, adapter):
        """Test deleting a collection."""
        docs = [Document(page_content="To be deleted", metadata={})]
        emb = [[0.1] * 384]
        await adapter.add_documents(docs, emb, book_id="deleteme", collection_id="Temp")

        collections = await adapter.list_collections()
        assert len(collections) == 1

        result = await adapter.delete_collection("Temp")
        assert result is True

        collections = await adapter.list_collections()
        assert collections == []

    async def test_delete_collection_not_found(self, adapter):
        """Test deleting a non-existent collection."""
        result = await adapter.delete_collection("nonexistent")
        assert result is False


class TestPostgresAdapterSearch:
    """Tests for hybrid search functionality."""

    async def test_vector_search(self, adapter):
        """Test vector similarity search."""
        docs = [
            Document(page_content="Python programming language", metadata={"page": 1}),
            Document(page_content="Java programming language", metadata={"page": 2}),
            Document(page_content="Cooking recipes", metadata={"page": 3}),
        ]
        py_embedding = [0.9, 0.1, 0.0] + [0.0] * 381
        java_embedding = [0.85, 0.15, 0.0] + [0.0] * 381
        cooking_embedding = [0.0, 0.0, 0.9] + [0.0] * 381
        embeddings = [py_embedding, java_embedding, cooking_embedding]

        await adapter.add_documents(docs, embeddings, book_id="searchtest")

        results = await adapter.similarity_search(
            query_embedding=py_embedding,
            k=2,
            query_text="Python",
        )

        assert len(results) >= 1
        assert "Python" in results[0].page_content

    async def test_text_search(self, adapter):
        """Test full-text search."""
        docs = [
            Document(page_content="The quick brown fox jumps", metadata={}),
            Document(page_content="A fast red fox running", metadata={}),
            Document(page_content="Cooking pasta al dente", metadata={}),
        ]
        embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]

        await adapter.add_documents(docs, embeddings, book_id="texttest")

        results = await adapter.similarity_search(
            query_embedding=[0.0] * 384,
            k=10,
            query_text="fox",
        )

        assert len(results) >= 2

    async def test_hybrid_search_with_filter(self, adapter):
        """Test hybrid search with book filter."""
        docs1 = [Document(page_content="Python code", metadata={})]
        emb1 = [[0.5] * 384]
        await adapter.add_documents(docs1, emb1, book_id="filter1", collection_id="Tech")

        docs2 = [Document(page_content="Python recipes", metadata={})]
        emb2 = [[0.6] * 384]
        await adapter.add_documents(docs2, emb2, book_id="filter2", collection_id="Cooking")

        results = await adapter.similarity_search(
            query_embedding=[0.5] * 384,
            k=10,
            collection_id="Tech",
            query_text="Python",
        )

        assert len(results) == 1
        assert results[0].metadata["book_id"] == "filter1"

    async def test_search_empty_results(self, adapter):
        """Test search with no results."""
        await adapter.clear()
        results = await adapter.similarity_search(
            query_embedding=[0.0] * 384,
            k=10,
            query_text="nonexistent content xyz123",
        )
        assert results == []


class TestPostgresAdapterBookOperations:
    """Tests for book-related operations."""

    async def test_get_book_chunk_count(self, adapter):
        """Test getting chunk count for a book."""
        docs = [
            Document(page_content=f"Page {i}", metadata={}) for i in range(5)
        ]
        embeddings = [[0.1 + i * 0.01] * 384 for i in range(5)]

        await adapter.add_documents(docs, embeddings, book_id="counttest")

        count = await adapter.get_book_chunk_count("counttest")
        assert count == 5

    async def test_get_book_chunk_count_not_found(self, adapter):
        """Test getting chunk count for non-existent book."""
        count = await adapter.get_book_chunk_count("nonexistentbook")
        assert count == 0

    async def test_book_metadata_preserved(self, adapter):
        """Test that book metadata is preserved."""
        docs = [
            Document(
                page_content="Test content",
                metadata={"title": "My Book", "author": "Test Author"}
            )
        ]
        embeddings = [[0.1] * 384]

        await adapter.add_documents(docs, embeddings, book_id="metadatatest")

        books = await adapter.list_books()
        book = next(b for b in books if b["id"] == "metadatatest")

        assert book["title"] == "My Book"
        assert book["author"] == "Test Author"


# Run with: pytest tests/integration/test_postgres_adapter.py -v
