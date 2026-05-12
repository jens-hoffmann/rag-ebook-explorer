"""Vector store adapters module."""

from ebook_rag_explorer.adapters.vectorstore.postgres_adapter import (
    PostgresAdapter,
    normalize_collection_id,
)

__all__ = ["PostgresAdapter", "normalize_collection_id"]
