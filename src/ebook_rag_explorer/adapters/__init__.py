"""Adapters module."""

from ebook_rag_explorer.adapters.embedding.sentence_transformer_adapter import (
    SentenceTransformerAdapter,
)
from ebook_rag_explorer.adapters.llm.langchain_llm_adapter import LangChainLLMAdapter
from ebook_rag_explorer.adapters.parsers import EpubParser, PdfParser
from ebook_rag_explorer.adapters.retrieval.cross_encoder_reranker import (
    CrossEncoderReranker,
)
from ebook_rag_explorer.adapters.retrieval.postgres_retriever import PostgresRetriever
from ebook_rag_explorer.adapters.vectorstore.postgres_adapter import (
    PostgresAdapter,
    normalize_collection_id,
)

__all__ = [
    "PdfParser",
    "EpubParser",
    "PostgresAdapter",
    "normalize_collection_id",
    "SentenceTransformerAdapter",
    "PostgresRetriever",
    "CrossEncoderReranker",
    "LangChainLLMAdapter",
]
