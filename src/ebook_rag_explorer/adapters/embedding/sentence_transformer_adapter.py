"""Sentence Transformers embedder adapter."""

from langchain_huggingface import HuggingFaceEmbeddings

from ebook_rag_explorer.ports.embedder_port import Embedder


class SentenceTransformerAdapter(Embedder):
    """Sentence Transformers implementation of the Embedder port."""

    def __init__(self, model_name: str) -> None:
        """Initialize the embedder with a sentence-transformers model.

        Args:
            model_name: Name of the sentence-transformers model.
        """
        self._model_name = model_name
        self._embedding_model: HuggingFaceEmbeddings | None = None
        self._embedding_dimension: int | None = None

    def _get_model(self) -> HuggingFaceEmbeddings:
        """Lazy load the embedding model.

        Returns:
            The HuggingFaceEmbeddings instance.
        """
        if self._embedding_model is None:
            self._embedding_model = HuggingFaceEmbeddings(
                model_name=self._model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            # Cache the dimension on first use
            test_embedding = self._embedding_model.embed_query("test")
            self._embedding_dimension = len(test_embedding)

        return self._embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text documents.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one for each input text.
        """
        if not texts:
            return []

        model = self._get_model()
        return model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: The query text to embed.

        Returns:
            The embedding vector for the query.
        """
        model = self._get_model()
        return model.embed_query(text)

    @property
    def embedding_dimension(self) -> int:
        """Return the dimensionality of the embeddings.

        Returns:
            The size of the embedding vectors.
        """
        if self._embedding_dimension is None:
            # Force model initialization
            _ = self._get_model()

        return self._embedding_dimension or 0

    @property
    def model_name(self) -> str:
        """Return the name of the embedding model.

        Returns:
            The model identifier string.
        """
        return self._model_name
