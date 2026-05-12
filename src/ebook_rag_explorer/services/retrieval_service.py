"""Retrieval service for orchestrating the RAG pipeline."""

from ebook_rag_explorer.models import SearchResponse, SourceDocument
from ebook_rag_explorer.ports.llm_port import LLM
from ebook_rag_explorer.ports.reranker_port import Reranker
from ebook_rag_explorer.ports.retriever_port import Retriever


class RetrievalService:
    """Service for orchestrating retrieval, reranking, and generation."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        llm: LLM,
        retrieval_top_k: int = 20,
        rerank_top_n: int = 5,
    ) -> None:
        """Initialize the retrieval service.

        Args:
            retriever: The retriever to fetch relevant documents.
            reranker: The reranker to reorder documents by relevance.
            llm: The LLM to generate answers.
            retrieval_top_k: Number of documents to retrieve initially.
            rerank_top_n: Number of documents after reranking.
        """
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_n = rerank_top_n

    def _format_context(self, documents: list) -> str:
        """Format documents into a context string for the LLM.

        Args:
            documents: List of documents to format.

        Returns:
            Formatted context string.
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source_info = []
            if doc.metadata.get("page"):
                source_info.append(f"page {doc.metadata['page']}")
            if doc.metadata.get("chapter"):
                source_info.append(f"chapter {doc.metadata['chapter']}")
            if doc.metadata.get("title"):
                source_info.append(f"\"{doc.metadata['title']}\"")

            source_str = f" [{', '.join(source_info)}]" if source_info else ""

            context_parts.append(
                f"[Document {i}{source_str}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        book_id: str | None = None,
        collection_id: str | None = None,
    ) -> SearchResponse:
        """Execute the full RAG search pipeline.

        This orchestrates: retrieve → rerank → generate answer.

        Args:
            query: The search query.
            top_k: Optional override for number of documents to retrieve.
            book_id: Optional filter to search within a specific book.
            collection_id: Optional filter to search within a specific collection.

        Returns:
            SearchResponse with answer and sources.
        """
        retrieval_k = top_k or self.retrieval_top_k

        # Step 1: Retrieve (hybrid search)
        retrieved_docs = await self.retriever.retrieve(
            query=query,
            top_k=retrieval_k,
            book_id=book_id,
            collection_id=collection_id,
        )

        if not retrieved_docs:
            return SearchResponse(
                query=query,
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                retrieved_count=0,
                reranked_count=0,
            )

        # Step 2: Rerank
        reranked_docs = self.reranker.rerank(
            query=query,
            documents=retrieved_docs,
            top_n=min(self.rerank_top_n, len(retrieved_docs)),
        )

        # Step 3: Generate answer
        context = self._format_context(reranked_docs)

        if self.llm.is_available:
            try:
                answer = self.llm.generate(query, context)
            except Exception:
                # Fallback if LLM fails
                answer = (
                    "I found some relevant information but couldn't generate a full answer. "
                    "Here are the most relevant passages:"
                )
        else:
            answer = (
                "LLM not configured. Here are the most relevant passages "
                "from the indexed documents:"
            )

        # Build source documents
        sources = [
            SourceDocument(
                content=doc.page_content,
                score=doc.metadata.get("rerank_score", doc.metadata.get("score", 0.0)),
                metadata={
                    k: v for k, v in doc.metadata.items()
                    if k not in ("rerank_score", "score", "rrf_score")
                },
            )
            for doc in reranked_docs
        ]

        return SearchResponse(
            query=query,
            answer=answer,
            sources=sources,
            retrieved_count=len(retrieved_docs),
            reranked_count=len(reranked_docs),
        )

    async def asearch(
        self,
        query: str,
        top_k: int | None = None,
        book_id: str | None = None,
        collection_id: str | None = None,
    ) -> SearchResponse:
        """Asynchronously execute the full RAG search pipeline.

        Args:
            query: The search query.
            top_k: Optional override for number of documents to retrieve.
            book_id: Optional filter to search within a specific book.
            collection_id: Optional filter to search within a specific collection.

        Returns:
            SearchResponse with answer and sources.
        """
        return await self.search(query, top_k, book_id, collection_id)
