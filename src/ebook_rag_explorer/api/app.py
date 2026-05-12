"""FastAPI application factory with lifespan management."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ebook_rag_explorer.adapters.embedding.sentence_transformer_adapter import (
    SentenceTransformerAdapter,
)
from ebook_rag_explorer.adapters.llm.langchain_llm_adapter import LangChainLLMAdapter
from ebook_rag_explorer.adapters.retrieval.chroma_retriever import ChromaRetriever
from ebook_rag_explorer.adapters.retrieval.cross_encoder_reranker import (
    CrossEncoderReranker,
)
from ebook_rag_explorer.adapters.vectorstore.chroma_adapter import ChromaAdapter
from ebook_rag_explorer.api.dependencies import set_embedder, set_retrieval_service, set_vector_store
from ebook_rag_explorer.config import Settings, get_settings
from ebook_rag_explorer.services.chunking_service import ChunkingService
from ebook_rag_explorer.services.indexing_service import IndexingService
from ebook_rag_explorer.services.retrieval_service import RetrievalService


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings instance. If not provided, will load from env.

    Returns:
        Configured FastAPI application.
    """
    if settings is None:
        settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan handler - initialize services on startup."""
        # Initialize on startup
        # Vector store
        vector_store = ChromaAdapter(settings.chroma_persist_path)
        set_vector_store(vector_store)

        # Embedder
        embedder = SentenceTransformerAdapter(settings.embedding_model)
        set_embedder(embedder)

        # Retriever
        retriever = ChromaRetriever(vector_store, embedder)

        # Reranker
        reranker = CrossEncoderReranker(settings.reranker_model)

        # LLM
        llm = LangChainLLMAdapter(
            provider=settings.llm_provider,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

        # Retrieval service
        retrieval_service = RetrievalService(
            retriever=retriever,
            reranker=reranker,
            llm=llm,
            retrieval_top_k=settings.retrieval_top_k,
            rerank_top_n=settings.rerank_top_n,
        )
        set_retrieval_service(retrieval_service)

        yield

        # Cleanup on shutdown (if needed)
        pass

    app = FastAPI(
        title="Ebook RAG Explorer API",
        description="RAG API for indexing and searching EPUB/PDF ebooks",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and include routers
    from ebook_rag_explorer.api.routes import books, collections, index, search

    app.include_router(index.router, prefix="/api", tags=["indexing"])
    app.include_router(search.router, prefix="/api", tags=["search"])
    app.include_router(books.router, prefix="/api", tags=["books"])
    app.include_router(collections.router, prefix="/api", tags=["collections"])

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


def main():
    """Main entry point for running the API server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "ebook_rag_explorer.api.app:create_app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    main()
