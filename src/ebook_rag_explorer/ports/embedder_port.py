"""Embedder port - abstract interface for text embedding models."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Protocol for text embedding implementations.

    Implementations should provide methods for embedding text documents
    using sentence-transformers or similar models.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text documents.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one for each input text.

        Raises:
            RuntimeError: If embedding fails.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: The query text to embed.

        Returns:
            The embedding vector for the query.

        Raises:
            RuntimeError: If embedding fails.
        """
        ...

    @property
    def embedding_dimension(self) -> int:
        """Return the dimensionality of the embeddings.

        Returns:
            The size of the embedding vectors.
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the name of the embedding model.

        Returns:
            The model identifier string.
        """
        ...
