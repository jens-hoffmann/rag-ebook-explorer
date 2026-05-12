"""API client for the Ebook RAG Explorer backend."""

import httpx
from dataclasses import dataclass
from typing import Optional

from ebook_rag_explorer.frontend.config import config


@dataclass
class SourceDocument:
    """Represents a source document from search results."""
    content: str
    score: float
    metadata: dict


@dataclass
class SearchResponse:
    """Search response from the API."""
    query: str
    answer: str
    sources: list[SourceDocument]
    retrieved_count: int
    reranked_count: int


@dataclass
class BookInfo:
    """Book information."""
    id: str
    title: Optional[str]
    format: str
    chunk_count: int
    collection_id: Optional[str]
    metadata: dict


@dataclass
class CollectionInfo:
    """Collection information."""
    id: str
    name: str
    book_count: int
    chunk_count: int


@dataclass
class IndexResponse:
    """Index response after uploading a book."""
    book_id: str
    title: Optional[str]
    chunks_indexed: int
    format: str
    message: str


class EbookRAGClient:
    """Async API client for the Ebook RAG Explorer backend."""

    def __init__(self, base_url: str = None):
        """Initialize the API client.

        Args:
            base_url: Base URL of the API. Defaults to config.default_api_url.
        """
        self.base_url = base_url or config.default_api_url
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(30.0),
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> dict:
        """Check if the API is healthy."""
        client = await self._get_client()
        response = await client.get("/health")
        response.raise_for_status()
        return response.json()

    async def search(
        self,
        query: str,
        collection_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> SearchResponse:
        """Search for documents.

        Args:
            query: The search query.
            collection_id: Optional collection to filter by.
            top_k: Optional override for number of results.

        Returns:
            SearchResponse with answer and sources.
        """
        client = await self._get_client()

        payload = {"query": query}
        if collection_id:
            payload["collection_id"] = collection_id
        if top_k:
            payload["top_k"] = top_k

        response = await client.post("/api/search", json=payload)
        response.raise_for_status()
        data = response.json()

        sources = [
            SourceDocument(
                content=src["content"],
                score=src["score"],
                metadata=src.get("metadata", {}),
            )
            for src in data.get("sources", [])
        ]

        return SearchResponse(
            query=data["query"],
            answer=data.get("answer", ""),
            sources=sources,
            retrieved_count=data.get("retrieved_count", 0),
            reranked_count=data.get("reranked_count", 0),
        )

    async def list_books(self, collection_id: Optional[str] = None) -> list[BookInfo]:
        """List all books.

        Args:
            collection_id: Optional collection to filter by.

        Returns:
            List of BookInfo objects.
        """
        client = await self._get_client()

        params = {}
        if collection_id:
            params["collection_id"] = collection_id

        response = await client.get("/api/books", params=params)
        response.raise_for_status()
        data = response.json()

        return [
            BookInfo(
                id=book["id"],
                title=book.get("title"),
                format=book.get("format", ""),
                chunk_count=book.get("chunk_count", 0),
                collection_id=book.get("collection_id"),
                metadata=book.get("metadata", {}),
            )
            for book in data
        ]

    async def delete_book(self, book_id: str) -> bool:
        """Delete a book.

        Args:
            book_id: The ID of the book to delete.

        Returns:
            True if successful, False otherwise.
        """
        client = await self._get_client()
        response = await client.delete(f"/api/books/{book_id}")
        return response.status_code == 204

    async def list_collections(self) -> list[CollectionInfo]:
        """List all collections.

        Returns:
            List of CollectionInfo objects.
        """
        client = await self._get_client()
        response = await client.get("/api/collections")
        response.raise_for_status()
        data = response.json()

        return [
            CollectionInfo(
                id=coll["id"],
                name=coll.get("name", coll["id"]),
                book_count=coll.get("book_count", 0),
                chunk_count=coll.get("chunk_count", 0),
            )
            for coll in data
        ]

    async def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection.

        Args:
            collection_id: The ID of the collection to delete.

        Returns:
            True if successful, False otherwise.
        """
        client = await self._get_client()
        response = await client.delete(f"/api/collections/{collection_id}")
        return response.status_code == 204

    async def index_file(
        self,
        file_bytes: bytes,
        filename: str,
        collection_id: Optional[str] = None,
        book_id: Optional[str] = None,
    ) -> IndexResponse:
        """Index a PDF or EPUB file.

        Args:
            file_bytes: The file content as bytes.
            filename: The original filename.
            collection_id: Optional collection to add the book to.
            book_id: Optional custom book ID.

        Returns:
            IndexResponse with indexing results.
        """
        client = await self._get_client()

        files = {"file": (filename, file_bytes)}
        data = {}
        if collection_id:
            data["collection_id"] = collection_id
        if book_id:
            data["book_id"] = book_id

        response = await client.post("/api/index", files=files, data=data)
        response.raise_for_status()
        result = response.json()

        return IndexResponse(
            book_id=result["book_id"],
            title=result.get("title"),
            chunks_indexed=result.get("chunks_indexed", 0),
            format=result.get("format", ""),
            message=result.get("message", ""),
        )


async def create_client(api_url: str) -> EbookRAGClient:
    """Create and return a new API client.

    Args:
        api_url: The base URL of the API.

    Returns:
        A new EbookRAGClient instance.
    """
    return EbookRAGClient(base_url=api_url)
