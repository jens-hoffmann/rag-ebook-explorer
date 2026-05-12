"""Vector store port - abstract interface for vector databases."""

from typing import Protocol, runtime_checkable

from langchain_core.documents import Document


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector store implementations.

    Implementations should provide methods for storing, retrieving, and managing
    document embeddings in a vector database.
    """

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        book_id: str,
    ) -> None:
        """Add documents with their embeddings to the vector store.

        Args:
            documents: List of Document objects to store.
            embeddings: List of embedding vectors corresponding to documents.
            book_id: Unique identifier for the book these documents belong to.

        Raises:
            ValueError: If documents and embeddings have different lengths.
            RuntimeError: If storing fails.
        """
        ...

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int,
        book_id: str | None = None,
    ) -> list[Document]:
        """Search for documents similar to the query embedding.

        Args:
            query_embedding: The query vector to search for.
            k: Number of results to return.
            book_id: Optional filter to search within a specific book.

        Returns:
            List of similar Document objects, ordered by relevance.
        """
        ...

    def delete_book(self, book_id: str) -> bool:
        """Delete all documents belonging to a specific book.

        Args:
            book_id: Unique identifier for the book to delete.

        Returns:
            True if the book was found and deleted, False otherwise.
        """
        ...

    def list_books(self) -> list[dict]:
        """List all indexed books with their metadata.

        Returns:
            List of dictionaries containing book info (id, title, chunk_count, etc.).
        """
        ...

    def get_book_chunk_count(self, book_id: str) -> int:
        """Get the number of chunks indexed for a specific book.

        Args:
            book_id: Unique identifier for the book.

        Returns:
            Number of chunks for the book, or 0 if not found.
        """
        ...

    def clear(self) -> None:
        """Clear all documents from the vector store (use with caution).

        This is primarily intended for testing purposes.
        """
        ...
