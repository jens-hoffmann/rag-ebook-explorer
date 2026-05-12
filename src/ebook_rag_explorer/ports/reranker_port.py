"""Reranker port - abstract interface for reranking retrieved documents."""

from typing import Protocol, runtime_checkable

from langchain_core.documents import Document


@runtime_checkable
class Reranker(Protocol):
    """Protocol for document reranking implementations.

    Implementations should use cross-encoders or similar models to
    rerank retrieved documents based on their relevance to a query.
    """

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int,
    ) -> list[Document]:
        """Rerank documents based on relevance to the query.

        Args:
            query: The search query text.
            documents: List of documents to rerank.
            top_n: Number of top documents to return after reranking.

        Returns:
            List of reranked Document objects, ordered by relevance score.
            The metadata of each document should include a 'rerank_score' key.

        Raises:
            RuntimeError: If reranking fails.
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the name of the reranking model.

        Returns:
            The model identifier string.
        """
        ...
