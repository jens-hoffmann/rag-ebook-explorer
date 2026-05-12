"""Shared fixtures for integration tests."""

import re
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from ebook_rag_explorer.adapters.vectorstore.postgres_adapter import PostgresAdapter


@pytest.fixture(scope="module")
def postgres_container():
    """Create a PostgreSQL container with pgvector."""
    with PostgresContainer("docker.io/pgvector/pgvector:pg16") as postgres:
        postgres.exec(f"psql -U {postgres.username} -d {postgres.dbname} -c 'CREATE EXTENSION IF NOT EXISTS vector;'")
        yield postgres


@pytest.fixture
def db_url(postgres_container) -> str:
    """Get the async database URL."""
    sync_url = postgres_container.get_connection_url()
    match = re.match(r"postgresql\+psycopg2://(.+):(.+)@(.+):(\d+)/(.+)", sync_url)
    if match:
        user, password, host, port, db = match.groups()
        async_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    else:
        async_url = sync_url.replace("postgresql+psycopg2://", "postgresql://")
    return async_url


@pytest_asyncio.fixture
async def adapter(db_url: str) -> AsyncGenerator[PostgresAdapter, None]:
    """Create a PostgresAdapter instance and initialize schema."""
    adapter = PostgresAdapter(db_url)

    pool = await adapter._get_pool()
    async with pool.acquire() as conn:
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

        await conn.execute("CREATE INDEX IF NOT EXISTS idx_books_collection ON books(collection_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_book_id ON documents(book_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)")

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

    await adapter.close()
