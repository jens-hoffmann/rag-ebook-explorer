"""Retriever port - abstract interface for document retrieval."""

from typing import Protocol, runtime_checkable

from langchain_core.documents import Document


@runtime_checkable
class Retriever(Protocol):
    """Protocol for document retrieval implementations.

    Implementations should provide semantic search capabilities over
    a vector store of embedded documents.
    """

    def retrieve(
        self,
        query: str,
        top_k: int,
        book_id: str | None = None,
    ) -> list[Document]:
        """Retrieve documents relevant to the query.

        Args:
            query: The search query text.
            top_k: Number of documents to retrieve.
            book_id: Optional filter to search within a specific book.

        Returns:
            List of relevant Document objects, ordered by relevance.

        Raises:
            RuntimeError: If retrieval fails.
        """
        ...

    async def aretrieve(
        self,
        query: str,
        top_k: int,
        book_id: str | None = None,
    ) -> list[Document]:
        """Asynchronously retrieve documents relevant to the query.

        Args:
            query: The search query text.
            top_k: Number of documents to retrieve.
            book_id: Optional filter to search within a specific book.

        Returns:
            List of relevant Document objects, ordered by relevance.
        """
        ...
