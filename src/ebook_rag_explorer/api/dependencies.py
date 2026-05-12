"""Dependency injection providers for FastAPI."""

from fastapi import Request

from ebook_rag_explorer.ports.embedder_port import Embedder
from ebook_rag_explorer.ports.vectorstore_port import VectorStore
from ebook_rag_explorer.services.indexing_service import IndexingService
from ebook_rag_explorer.services.retrieval_service import RetrievalService

# Global instances (set during app lifespan)
_vector_store: VectorStore | None = None
_embedder: Embedder | None = None
_retrieval_service: RetrievalService | None = None


def set_vector_store(store: VectorStore) -> None:
    """Set the global vector store instance."""
    global _vector_store
    _vector_store = store


def set_embedder(embedder: Embedder) -> None:
    """Set the global embedder instance."""
    global _embedder
    _embedder = embedder


def set_retrieval_service(service: RetrievalService) -> None:
    """Set the global retrieval service instance."""
    global _retrieval_service
    _retrieval_service = service


def get_vector_store() -> VectorStore:
    """Get the vector store instance.

    Returns:
        The configured vector store.

    Raises:
        RuntimeError: If vector store hasn't been initialized.
    """
    if _vector_store is None:
        raise RuntimeError("Vector store not initialized")
    return _vector_store


def get_embedder() -> Embedder:
    """Get the embedder instance.

    Returns:
        The configured embedder.

    Raises:
        RuntimeError: If embedder hasn't been initialized.
    """
    if _embedder is None:
        raise RuntimeError("Embedder not initialized")
    return _embedder


def get_retrieval_service() -> RetrievalService:
    """Get the retrieval service instance.

    Returns:
        The configured retrieval service.

    Raises:
        RuntimeError: If retrieval service hasn't been initialized.
    """
    if _retrieval_service is None:
        raise RuntimeError("Retrieval service not initialized")
    return _retrieval_service


def get_indexing_service(request: Request) -> IndexingService:
    """Create an indexing service for the current request.

    This creates a fresh indexing service with the configured dependencies.

    Args:
        request: The FastAPI request object.

    Returns:
        A new IndexingService instance.
    """
    from ebook_rag_explorer.config import get_settings
    from ebook_rag_explorer.services.chunking_service import ChunkingService

    settings = get_settings()

    vector_store = get_vector_store()
    embedder = get_embedder()
    chunking_service = ChunkingService(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return IndexingService(
        vector_store=vector_store,
        embedder=embedder,
        chunking_service=chunking_service,
    )
