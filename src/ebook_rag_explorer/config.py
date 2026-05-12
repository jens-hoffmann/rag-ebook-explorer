"""Application configuration using Pydantic BaseSettings."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Embedding Configuration
    # -------------------------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    """Sentence-transformers compatible model for document embeddings."""

    embedding_dimension: int = 384
    """Dimension of the embedding vectors (384 for all-MiniLM-L6-v2)."""

    # -------------------------------------------------------------------------
    # Reranking Configuration
    # -------------------------------------------------------------------------
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    """CrossEncoder compatible model for reranking retrieved documents."""

    # -------------------------------------------------------------------------
    # LLM Configuration
    # -------------------------------------------------------------------------
    llm_provider: Literal["openai", "lmstudio", "azure"] = "openai"
    """LLM provider: openai, lmstudio, or azure."""

    llm_model: str = "gpt-4o-mini"
    """Model name or deployment name for the LLM."""

    llm_base_url: str | None = None
    """Optional base URL for the LLM API (e.g., http://localhost:1234/v1 for LM Studio)."""

    llm_api_key: str | None = None
    """API key for OpenAI or Azure OpenAI."""

    # -------------------------------------------------------------------------
    # Database Configuration (PostgreSQL + pgvector)
    # -------------------------------------------------------------------------
    database_url: str = "postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb"
    """Database connection URL (asyncpg driver)."""

    postgres_host: str = "localhost"
    """PostgreSQL host."""

    postgres_port: int = 5432
    """PostgreSQL port."""

    postgres_user: str = "raguser"
    """PostgreSQL username."""

    postgres_password: str = "ragpassword"
    """PostgreSQL password."""

    postgres_db: str = "ragdb"
    """PostgreSQL database name."""

    # -------------------------------------------------------------------------
    # Hybrid Search Configuration
    # -------------------------------------------------------------------------
    retrieval_top_k: int = 20
    """Number of documents to retrieve via vector similarity."""

    text_search_top_k: int = 20
    """Number of documents to retrieve via full-text search."""

    hybrid_top_k: int = 40
    """Total results to consider from both search methods before reranking."""

    rrf_k: int = 60
    """RRF (Reciprocal Rank Fusion) parameter. Higher = more weight to top results."""

    rerank_top_n: int = 5
    """Number of documents to return after reranking."""

    # -------------------------------------------------------------------------
    # Chunking Configuration
    # -------------------------------------------------------------------------
    chunk_size: int = 1000
    """Size of text chunks in characters."""

    chunk_overlap: int = 200
    """Overlap between consecutive chunks in characters."""

    # -------------------------------------------------------------------------
    # API Configuration
    # -------------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    """Host for the FastAPI server."""

    api_port: int = 8000
    """Port for the FastAPI server."""


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance (singleton pattern).

    Returns:
        Settings: The application settings.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance (useful for testing).

    This forces the next call to get_settings() to reload from environment.
    """
    global _settings
    _settings = None
