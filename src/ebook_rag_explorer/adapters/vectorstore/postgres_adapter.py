"""PostgreSQL + pgvector adapter with hybrid search."""

import json
import re
import uuid
from typing import Any

import asyncpg
from langchain_core.documents import Document

from ebook_rag_explorer.ports.vectorstore_port import VectorStore


def normalize_collection_id(name: str) -> str:
    """Normalize a collection name to a valid ID.

    Args:
        name: The collection name to normalize.

    Returns:
        Normalized collection ID (lowercase, stripped, special chars removed).
    """
    # Strip whitespace, lowercase, replace non-alphanumeric with underscore
    normalized = name.strip().lower()
    normalized = re.sub(r"[^a-z0-9_\-]", "_", normalized)
    # Remove consecutive underscores
    normalized = re.sub(r"_+", "_", normalized)
    # Strip leading/trailing underscores
    normalized = normalized.strip("_")
    return normalized or "default"


class PostgresAdapter(VectorStore):
    """PostgreSQL + pgvector implementation of the VectorStore port with hybrid search."""

    def __init__(self, database_url: str) -> None:
        """Initialize the PostgreSQL adapter.

        Args:
            database_url: Database connection URL (asyncpg format).
        """
        self.database_url = database_url
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create the connection pool.

        Returns:
            The asyncpg connection pool.
        """
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
            )
        return self._pool

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _format_vector(self, vector: list[float]) -> str:
        """Format a vector for PostgreSQL.

        Args:
            vector: The vector to format.

        Returns:
            PostgreSQL vector literal string.
        """
        return "[" + ",".join(str(x) for x in vector) + "]"

    async def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        book_id: str,
        collection_id: str | None = None,
    ) -> None:
        """Add documents with their embeddings to the database.

        Args:
            documents: List of Document objects to store.
            embeddings: List of embedding vectors corresponding to documents.
            book_id: Unique identifier for the book these documents belong to.
            collection_id: Optional collection to organize the book into.

        Raises:
            ValueError: If documents and embeddings have different lengths.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Documents ({len(documents)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )

        if not documents:
            return

        # Normalize collection_id
        normalized_collection = (
            normalize_collection_id(collection_id) if collection_id else None
        )

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Insert book if not exists
            await conn.execute(
                """
                INSERT INTO books (id, title, author, format, collection_id)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    collection_id = COALESCE(EXCLUDED.collection_id, books.collection_id)
                """,
                book_id,
                documents[0].metadata.get("title", ""),
                documents[0].metadata.get("author", ""),
                documents[0].metadata.get("format", ""),
                normalized_collection,
            )

            # Insert documents
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                chunk_id = f"{book_id}_{i}_{uuid.uuid4().hex[:8]}"

                await conn.execute(
                    """
                    INSERT INTO documents 
                    (chunk_id, book_id, collection_id, content, embedding, 
                     chunk_index, total_chunks, source_metadata)
                    VALUES ($1, $2, $3, $4, $5::vector, $6, $7, $8::jsonb)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        source_metadata = EXCLUDED.source_metadata
                    """,
                    chunk_id,
                    book_id,
                    normalized_collection,
                    doc.page_content,
                    self._format_vector(embedding),
                    i,
                    len(documents),
                    json.dumps(doc.metadata) if doc.metadata else "{}",
                )

            # Update book chunk count
            await conn.execute(
                """
                UPDATE books SET chunk_count = $1 WHERE id = $2
                """,
                len(documents),
                book_id,
            )

    async def _vector_search(
        self,
        query_embedding: list[float],
        k: int,
        book_id: str | None = None,
        collection_id: str | None = None,
    ) -> list[tuple[Document, float]]:
        """Perform vector similarity search.

        Args:
            query_embedding: The query vector.
            k: Number of results.
            book_id: Optional book filter.
            collection_id: Optional collection filter.

        Returns:
            List of (document, score) tuples.
        """
        pool = await self._get_pool()
        vector_str = self._format_vector(query_embedding)

        # Build where clause
        conditions = ["1=1"]
        params: list[Any] = [vector_str, k]
        param_idx = 3

        if book_id:
            conditions.append(f"book_id = ${param_idx}")
            params.append(book_id)
            param_idx += 1

        if collection_id:
            conditions.append(f"collection_id = ${param_idx}")
            params.append(normalize_collection_id(collection_id))
            param_idx += 1

        where_clause = " AND ".join(conditions)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT 
                    chunk_id, book_id, collection_id, content,
                    chunk_index, total_chunks, source_metadata,
                    1 - (embedding <=> $1::vector) as score
                FROM documents
                WHERE {where_clause}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                *params,
            )

        results = []
        for row in rows:
            metadata = row["source_metadata"] if isinstance(row["source_metadata"], dict) else {}
            metadata.update({
                "chunk_id": row["chunk_id"],
                "book_id": row["book_id"],
                "collection_id": row["collection_id"],
                "chunk_index": row["chunk_index"],
                "total_chunks": row["total_chunks"],
                "score": row["score"],
            })

            doc = Document(
                page_content=row["content"],
                metadata=metadata,
            )
            results.append((doc, row["score"]))

        return results

    async def _text_search(
        self,
        query: str,
        k: int,
        book_id: str | None = None,
        collection_id: str | None = None,
    ) -> list[tuple[Document, float]]:
        """Perform full-text search.

        Args:
            query: The search query text.
            k: Number of results.
            book_id: Optional book filter.
            collection_id: Optional collection filter.

        Returns:
            List of (document, score) tuples.
        """
        pool = await self._get_pool()

        # Build where clause
        conditions = ["search_vector @@ plainto_tsquery('english', $1)"]
        params: list[Any] = [query, k]
        param_idx = 3

        if book_id:
            conditions.append(f"book_id = ${param_idx}")
            params.append(book_id)
            param_idx += 1

        if collection_id:
            conditions.append(f"collection_id = ${param_idx}")
            params.append(normalize_collection_id(collection_id))
            param_idx += 1

        where_clause = " AND ".join(conditions)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT 
                    chunk_id, book_id, collection_id, content,
                    chunk_index, total_chunks, source_metadata,
                    ts_rank_cd(search_vector, plainto_tsquery('english', $1), 32) as score
                FROM documents
                WHERE {where_clause}
                ORDER BY score DESC
                LIMIT $2
                """,
                *params,
            )

        results = []
        for row in rows:
            metadata = row["source_metadata"] if isinstance(row["source_metadata"], dict) else {}
            metadata.update({
                "chunk_id": row["chunk_id"],
                "book_id": row["book_id"],
                "collection_id": row["collection_id"],
                "chunk_index": row["chunk_index"],
                "total_chunks": row["total_chunks"],
                "text_score": row["score"],
            })

            doc = Document(
                page_content=row["content"],
                metadata=metadata,
            )
            results.append((doc, row["score"]))

        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[tuple[Document, float]],
        text_results: list[tuple[Document, float]],
        k: int = 60,
    ) -> list[Document]:
        """Combine results using Reciprocal Rank Fusion (RRF).

        RRF score = Σ(1/(k + rank)) for each result set.

        Args:
            vector_results: Results from vector search with scores.
            text_results: Results from text search with scores.
            k: RRF parameter (default 60).

        Returns:
            Fused and reranked list of documents.
        """
        # Build score dictionary: chunk_id -> RRF score
        rrf_scores: dict[str, tuple[float, Document]] = {}

        # Process vector results
        for rank, (doc, _) in enumerate(vector_results, start=1):
            chunk_id = doc.metadata.get("chunk_id", "")
            score = 1.0 / (k + rank)
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] = (rrf_scores[chunk_id][0] + score, doc)
            else:
                rrf_scores[chunk_id] = (score, doc)

        # Process text results
        for rank, (doc, _) in enumerate(text_results, start=1):
            chunk_id = doc.metadata.get("chunk_id", "")
            score = 1.0 / (k + rank)
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] = (rrf_scores[chunk_id][0] + score, doc)
            else:
                rrf_scores[chunk_id] = (score, doc)

        # Sort by RRF score descending
        sorted_results = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)

        # Add RRF score to metadata
        documents = []
        for score, doc in sorted_results:
            doc.metadata["rrf_score"] = score
            documents.append(doc)

        return documents

    async def similarity_search(
        self,
        query_embedding: list[float],
        k: int,
        book_id: str | None = None,
        collection_id: str | None = None,
        query_text: str | None = None,
    ) -> list[Document]:
        """Hybrid search combining vector similarity and full-text search.

        Args:
            query_embedding: The query vector for semantic search.
            k: Number of results to return.
            book_id: Optional filter to search within a specific book.
            collection_id: Optional filter to search within a specific collection.
            query_text: Original query text for full-text search.

        Returns:
            List of similar Document objects, ordered by RRF score.
        """
        # If no text query provided, do vector-only search
        if not query_text:
            results = await self._vector_search(
                query_embedding, k, book_id, collection_id
            )
            return [doc for doc, _ in results]

        # Perform both searches
        vector_results = await self._vector_search(
            query_embedding, k * 2, book_id, collection_id
        )
        text_results = await self._text_search(
            query_text, k * 2, book_id, collection_id
        )

        # Fuse results using RRF
        fused = self._reciprocal_rank_fusion(vector_results, text_results)

        return fused[:k]

    async def delete_book(self, book_id: str) -> bool:
        """Delete all documents belonging to a specific book.

        Args:
            book_id: Unique identifier for the book to delete.

        Returns:
            True if the book was found and deleted, False otherwise.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Check if book exists
            row = await conn.fetchrow(
                "SELECT id FROM books WHERE id = $1",
                book_id,
            )

            if not row:
                return False

            # Delete book (cascades to documents via FK)
            await conn.execute(
                "DELETE FROM books WHERE id = $1",
                book_id,
            )

            return True

    async def list_books(self) -> list[dict]:
        """List all indexed books with their metadata.

        Returns:
            List of dictionaries containing book info.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, title, author, format, collection_id, 
                       metadata, chunk_count
                FROM books
                ORDER BY created_at DESC
                """
            )

        return [
            {
                "id": row["id"],
                "title": row["title"],
                "author": row["author"],
                "format": row["format"],
                "collection_id": row["collection_id"],
                "metadata": row["metadata"] if isinstance(row["metadata"], dict) else {},
                "chunk_count": row["chunk_count"],
            }
            for row in rows
        ]

    async def get_book_chunk_count(self, book_id: str) -> int:
        """Get the number of chunks indexed for a specific book.

        Args:
            book_id: Unique identifier for the book.

        Returns:
            Number of chunks for the book, or 0 if not found.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT chunk_count FROM books WHERE id = $1",
                book_id,
            )

            return row["chunk_count"] if row else 0

    async def list_collections(self) -> list[dict]:
        """List all collections with their metadata.

        Returns:
            List of dictionaries containing collection info.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, book_count, chunk_count
                FROM collection_stats
                """
            )

        return [
            {
                "id": row["id"],
                "name": row["name"],
                "book_count": row["book_count"],
                "chunk_count": row["chunk_count"],
            }
            for row in rows
        ]

    async def delete_collection(self, collection_id: str) -> bool:
        """Delete all documents belonging to a specific collection.

        Args:
            collection_id: Collection identifier to delete.

        Returns:
            True if the collection was found and deleted, False otherwise.
        """
        normalized = normalize_collection_id(collection_id)
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Check if collection exists
            row = await conn.fetchrow(
                "SELECT 1 FROM documents WHERE collection_id = $1 LIMIT 1",
                normalized,
            )

            if not row:
                return False

            # Delete all books in the collection (cascades to documents)
            await conn.execute(
                "DELETE FROM books WHERE collection_id = $1",
                normalized,
            )

            return True

    async def clear(self) -> None:
        """Clear all documents from the database."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM documents")
            await conn.execute("DELETE FROM books")
