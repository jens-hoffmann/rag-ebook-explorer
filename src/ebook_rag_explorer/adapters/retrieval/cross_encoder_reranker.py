"""CrossEncoder reranker adapter."""

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from ebook_rag_explorer.ports.reranker_port import Reranker


class CrossEncoderReranker(Reranker):
    """CrossEncoder implementation of the Reranker port."""

    def __init__(self, model_name: str) -> None:
        """Initialize the reranker with a cross-encoder model.

        Args:
            model_name: Name of the cross-encoder model from sentence-transformers.
        """
        self._model_name = model_name
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        """Lazy load the cross-encoder model.

        Returns:
            The CrossEncoder instance.
        """
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

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
            List of reranked Document objects with rerank_score in metadata.
        """
        if not documents:
            return []

        if len(documents) <= top_n:
            # Add scores even if we don't need to filter
            model = self._get_model()
            pairs = [[query, doc.page_content] for doc in documents]
            scores = model.predict(pairs)

            for doc, score in zip(documents, scores):
                doc.metadata = {
                    **doc.metadata,
                    "rerank_score": float(score),
                }
            return documents

        # Rerank with cross-encoder
        model = self._get_model()

        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Get relevance scores
        scores = model.predict(pairs)

        # Create scored documents
        scored_docs = []
        for doc, score in zip(documents, scores):
            scored_doc = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "rerank_score": float(score),
                },
            )
            scored_docs.append((scored_doc, score))

        # Sort by score (descending) and take top_n
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_documents = [doc for doc, _ in scored_docs[:top_n]]

        return top_documents

    @property
    def model_name(self) -> str:
        """Return the name of the reranking model.

        Returns:
            The model identifier string.
        """
        return self._model_name
