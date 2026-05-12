"""PostgreSQL retriever adapter with hybrid search."""

from langchain_core.documents import Document

from ebook_rag_explorer.adapters.vectorstore.postgres_adapter import PostgresAdapter
from ebook_rag_explorer.ports.embedder_port import Embedder
from ebook_rag_explorer.ports.retriever_port import Retriever


class PostgresRetriever(Retriever):
    """PostgreSQL implementation of the Retriever port with hybrid search."""

    def __init__(self, vector_store: PostgresAdapter, embedder: Embedder) -> None:
        """Initialize the retriever.

        Args:
            vector_store: The PostgreSQL adapter.
            embedder: The embedder to encode queries.
        """
        self.vector_store = vector_store
        self.embedder = embedder

    async def retrieve(
        self,
        query: str,
        top_k: int,
        book_id: str | None = None,
        collection_id: str | None = None,
    ) -> list[Document]:
        """Retrieve documents using hybrid search (vector + text).

        Args:
            query: The search query text.
            top_k: Number of documents to retrieve.
            book_id: Optional filter to search within a specific book.
            collection_id: Optional filter to search within a specific collection.

        Returns:
            List of relevant Document objects, ordered by RRF score.
        """
        # Embed the query
        query_embedding = self.embedder.embed_query(query)

        # Perform hybrid search
        return await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=top_k,
            book_id=book_id,
            collection_id=collection_id,
            query_text=query,
        )

    async def aretrieve(
        self,
        query: str,
        top_k: int,
        book_id: str | None = None,
        collection_id: str | None = None,
    ) -> list[Document]:
        """Asynchronously retrieve documents (alias for retrieve).

        Args:
            query: The search query text.
            top_k: Number of documents to retrieve.
            book_id: Optional filter to search within a specific book.
            collection_id: Optional filter to search within a specific collection.

        Returns:
            List of relevant Document objects.
        """
        return await self.retrieve(query, top_k, book_id, collection_id)
