"""ChromaDB retriever adapter."""

from langchain_core.documents import Document

from ebook_rag_explorer.ports.embedder_port import Embedder
from ebook_rag_explorer.ports.retriever_port import Retriever
from ebook_rag_explorer.ports.vectorstore_port import VectorStore


class ChromaRetriever(Retriever):
    """ChromaDB implementation of the Retriever port."""

    def __init__(self, vector_store: VectorStore, embedder: Embedder) -> None:
        """Initialize the retriever.

        Args:
            vector_store: The vector store to search.
            embedder: The embedder to encode queries.
        """
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int,
        book_id: str | None = None,
    ) -> list[Document]:
        """Retrieve documents relevant to the query.

        Args:
            query: The search query text.
            top_k: Number of documents to retrieve.
            book_id: Optional filter to search within a specific book.

        Returns:
            List of relevant Document objects, ordered by relevance.
        """
        # Embed the query
        query_embedding = self.embedder.embed_query(query)

        # Search the vector store
        return self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=top_k,
            book_id=book_id,
        )

    async def aretrieve(
        self,
        query: str,
        top_k: int,
        book_id: str | None = None,
    ) -> list[Document]:
        """Asynchronously retrieve documents relevant to the query.

        Note: Currently delegates to sync method. Can be optimized with
        async embedder in the future.

        Args:
            query: The search query text.
            top_k: Number of documents to retrieve.
            book_id: Optional filter to search within a specific book.

        Returns:
            List of relevant Document objects, ordered by relevance.
        """
        # For now, delegate to sync method
        # TODO: Implement true async when embedder supports it
        return self.retrieve(query, top_k, book_id)
