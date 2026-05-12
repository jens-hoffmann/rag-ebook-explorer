"""Integration tests for PostgreSQL adapter using testcontainers."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from testcontainers.postgres import PostgresContainer

from ebook_rag_explorer.adapters.vectorstore.postgres_adapter import (
    PostgresAdapter,
    normalize_collection_id,
)


@pytest.fixture(scope="module")
def postgres_container():
    """Create a PostgreSQL container with pgvector."""
    with PostgresContainer("docker.io/pgvector/pgvector:pg16") as postgres:
        # Wait for database to be ready
        postgres.exec(f"psql -U {postgres.username} -d {postgres.dbname} -c 'CREATE EXTENSION IF NOT EXISTS vector;'")
        yield postgres


@pytest.fixture
async def db_url(postgres_container) -> str:
    """Get the async database URL."""
    # testcontainers returns postgresql+psycopg2:// but asyncpg needs postgresql://
    sync_url = postgres_container.get_connection_url()
    # Extract the parts we need: postgresql://user:pass@host:port/db
    import re
    match = re.match(r"postgresql\+psycopg2://(.+):(.+)@(.+):(\d+)/(.+)", sync_url)
    if match:
        user, password, host, port, db = match.groups()
        async_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    else:
        # Fallback: just replace the scheme
        async_url = sync_url.replace("postgresql+psycopg2://", "postgresql://")
    return async_url


@pytest.fixture
async def adapter(db_url: str) -> PostgresAdapter:
    """Create a PostgresAdapter instance and initialize schema."""
    adapter = PostgresAdapter(db_url)

    # Create tables using raw SQL since init script may not run in testcontainers
    pool = await adapter._get_pool()
    async with pool.acquire() as conn:
        # Create books table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS books (
                id VARCHAR(255) PRIMARY KEY,
                title VARCHAR(500),
                author VARCHAR(255),
                format VARCHAR(50),
                collection_id VARCHAR(255),
                metadata JSONB DEFAULT '{}',
                chunk_count INTEGER DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        # Create documents table with pgvector
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                chunk_id VARCHAR(255) NOT NULL UNIQUE,
                book_id VARCHAR(255) REFERENCES books(id) ON DELETE CASCADE,
                collection_id VARCHAR(255),
                content TEXT NOT NULL,
                embedding VECTOR(384),
                search_vector tsvector GENERATED ALWAYS AS (
                    setweight(to_tsvector('english', COALESCE(content, '')), 'A')
                ) STORED,
                chunk_index INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 0,
                source_metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        # Create indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_books_collection ON books(collection_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_book_id ON documents(book_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)")

        # Create collection_stats view
        await conn.execute("""
            CREATE OR REPLACE VIEW collection_stats AS
            SELECT
                collection_id AS id,
                collection_id AS name,
                COUNT(DISTINCT book_id) AS book_count,
                COUNT(*) AS chunk_count
            FROM documents
            WHERE collection_id IS NOT NULL
            GROUP BY collection_id
        """)

    yield adapter

    # Cleanup
    await adapter.close()


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

    async def test_add_documents(self, adapter: PostgresAdapter):
        """Test adding documents to the database."""
        docs = [
            Document(page_content="Python is a great programming language.", metadata={"page": 1}),
            Document(page_content="Machine learning uses Python extensively.", metadata={"page": 2}),
        ]
        embeddings = [
            [0.1, 0.2, 0.3] + [0.0] * 381,  # 384 dimensions
            [0.4, 0.5, 0.6] + [0.0] * 381,
        ]

        await adapter.add_documents(docs, embeddings, book_id="book1")

        books = await adapter.list_books()
        assert len(books) == 1
        assert books[0]["id"] == "book1"
        assert books[0]["chunk_count"] == 2

    async def test_add_documents_with_collection(self, adapter: PostgresAdapter):
        """Test adding documents with a collection."""
        docs = [Document(page_content="Test content", metadata={})]
        embeddings = [[0.1] * 384]

        await adapter.add_documents(docs, embeddings, book_id="book2", collection_id="My Collection")

        books = await adapter.list_books()
        assert books[0]["collection_id"] == "my_collection"

    async def test_delete_book(self, adapter: PostgresAdapter):
        """Test deleting a book."""
        docs = [Document(page_content="Content to delete", metadata={})]
        embeddings = [[0.1] * 384]

        await adapter.add_documents(docs, embeddings, book_id="book3")

        # Verify it exists
        books = await adapter.list_books()
        assert len(books) == 3  # Previous tests added 2

        # Delete
        result = await adapter.delete_book("book3")
        assert result is True

        # Verify it's gone
        books = await adapter.list_books()
        assert len(books) == 2

    async def test_delete_book_not_found(self, adapter: PostgresAdapter):
        """Test deleting a non-existent book."""
        result = await adapter.delete_book("nonexistent")
        assert result is False

    async def test_list_books_empty(self, adapter: PostgresAdapter):
        """Test listing books when none exist."""
        # Clear all books
        await adapter.clear()
        books = await adapter.list_books()
        assert books == []

    async def test_clear(self, adapter: PostgresAdapter):
        """Test clearing all documents."""
        docs = [Document(page_content="Content", metadata={})]
        embeddings = [[0.1] * 384]
        await adapter.add_documents(docs, embeddings, book_id="cleartest")

        await adapter.clear()

        books = await adapter.list_books()
        assert books == []


class TestPostgresAdapterCollections:
    """Tests for collection functionality."""

    async def test_list_collections_empty(self, adapter: PostgresAdapter):
        """Test listing collections when none exist."""
        await adapter.clear()
        collections = await adapter.list_collections()
        assert collections == []

    async def test_list_collections_with_data(self, adapter: PostgresAdapter):
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

        # Should have 2 collections
        assert len(collections) == 2

        # Find Programming collection
        prog = next(c for c in collections if c["id"] == "programming")
        assert prog["book_count"] == 2
        assert prog["chunk_count"] == 2

        # Find Cooking collection
        cooking = next(c for c in collections if c["id"] == "cooking")
        assert cooking["book_count"] == 1
        assert cooking["chunk_count"] == 1

        # Clean up for next test
        await adapter.clear()

    async def test_delete_collection(self, adapter: PostgresAdapter):
        """Test deleting a collection."""
        docs = [Document(page_content="To be deleted", metadata={})]
        emb = [[0.1] * 384]
        await adapter.add_documents(docs, emb, book_id="deleteme", collection_id="Temp")

        # Verify it exists
        collections = await adapter.list_collections()
        assert len(collections) == 1

        # Delete
        result = await adapter.delete_collection("Temp")
        assert result is True

        # Verify it's gone
        collections = await adapter.list_collections()
        assert collections == []

    async def test_delete_collection_not_found(self, adapter: PostgresAdapter):
        """Test deleting a non-existent collection."""
        result = await adapter.delete_collection("nonexistent")
        assert result is False


class TestPostgresAdapterSearch:
    """Tests for hybrid search functionality."""

    async def test_vector_search(self, adapter: PostgresAdapter):
        """Test vector similarity search."""
        docs = [
            Document(page_content="Python programming language", metadata={"page": 1}),
            Document(page_content="Java programming language", metadata={"page": 2}),
            Document(page_content="Cooking recipes", metadata={"page": 3}),
        ]
        # Create embeddings that are close for similar content
        py_embedding = [0.9, 0.1, 0.0] + [0.0] * 381
        java_embedding = [0.85, 0.15, 0.0] + [0.0] * 381
        cooking_embedding = [0.0, 0.0, 0.9] + [0.0] * 381
        embeddings = [py_embedding, java_embedding, cooking_embedding]

        await adapter.add_documents(docs, embeddings, book_id="searchtest")

        # Search for Python content
        results = await adapter.similarity_search(
            query_embedding=py_embedding,
            k=2,
            query_text="Python",  # Also use text search
        )

        assert len(results) >= 1
        # First result should be about Python
        assert "Python" in results[0].page_content

    async def test_text_search(self, adapter: PostgresAdapter):
        """Test full-text search."""
        docs = [
            Document(page_content="The quick brown fox jumps", metadata={}),
            Document(page_content="A fast red fox running", metadata={}),
            Document(page_content="Cooking pasta al dente", metadata={}),
        ]
        embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]

        await adapter.add_documents(docs, embeddings, book_id="texttest")

        # Search for "fox"
        results = await adapter.similarity_search(
            query_embedding=[0.0] * 384,
            k=10,
            query_text="fox",
        )

        assert len(results) >= 2  # Should find both fox documents

    async def test_hybrid_search_with_filter(self, adapter: PostgresAdapter):
        """Test hybrid search with book filter."""
        docs1 = [Document(page_content="Python code", metadata={})]
        emb1 = [[0.5] * 384]
        await adapter.add_documents(docs1, emb1, book_id="filter1", collection_id="Tech")

        docs2 = [Document(page_content="Python recipes", metadata={})]
        emb2 = [[0.6] * 384]
        await adapter.add_documents(docs2, emb2, book_id="filter2", collection_id="Cooking")

        # Search within Tech collection only
        results = await adapter.similarity_search(
            query_embedding=[0.5] * 384,
            k=10,
            collection_id="Tech",
            query_text="Python",
        )

        # Should only find filter1
        assert len(results) == 1
        assert results[0].metadata["book_id"] == "filter1"

    async def test_search_empty_results(self, adapter: PostgresAdapter):
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

    async def test_get_book_chunk_count(self, adapter: PostgresAdapter):
        """Test getting chunk count for a book."""
        docs = [
            Document(page_content=f"Page {i}", metadata={}) for i in range(5)
        ]
        embeddings = [[0.1 + i * 0.01] * 384 for i in range(5)]

        await adapter.add_documents(docs, embeddings, book_id="counttest")

        count = await adapter.get_book_chunk_count("counttest")
        assert count == 5

    async def test_get_book_chunk_count_not_found(self, adapter: PostgresAdapter):
        """Test getting chunk count for non-existent book."""
        count = await adapter.get_book_chunk_count("nonexistentbook")
        assert count == 0

    async def test_book_metadata_preserved(self, adapter: PostgresAdapter):
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
