"""Integration tests for ChromaDB adapter."""

import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from ebook_rag_explorer.adapters.vectorstore.chroma_adapter import ChromaAdapter


class TestChromaAdapter:
    """Integration tests for ChromaDB adapter using temporary storage."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for ChromaDB persistence."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def adapter(self, temp_persist_dir: Path) -> ChromaAdapter:
        """Create a ChromaAdapter instance with temp storage."""
        return ChromaAdapter(persist_dir=temp_persist_dir)

    @pytest.fixture
    def sample_documents(self) -> list[Document]:
        """Create sample documents for testing."""
        return [
            Document(
                page_content="This is the first document about Python programming.",
                metadata={"page": 1, "title": "Python Guide", "author": "Test Author"},
            ),
            Document(
                page_content="This is the second document about machine learning.",
                metadata={"page": 2, "title": "Python Guide", "author": "Test Author"},
            ),
            Document(
                page_content="This is the third document about data science.",
                metadata={"page": 3, "title": "Python Guide", "author": "Test Author"},
            ),
        ]

    @pytest.fixture
    def sample_embeddings(self) -> list[list[float]]:
        """Create sample embeddings (3 documents, 5 dimensions each)."""
        return [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]

    def test_add_documents_success(
        self,
        adapter: ChromaAdapter,
        sample_documents: list[Document],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test successfully adding documents to the vector store."""
        book_id = "test-book-1"

        adapter.add_documents(sample_documents, sample_embeddings, book_id)

        # Verify documents were added
        count = adapter.get_book_chunk_count(book_id)
        assert count == len(sample_documents)

    def test_add_documents_empty_list(self, adapter: ChromaAdapter) -> None:
        """Test adding an empty list of documents."""
        adapter.add_documents([], [], "empty-book")
        count = adapter.get_book_chunk_count("empty-book")
        assert count == 0

    def test_add_documents_mismatched_lengths(
        self,
        adapter: ChromaAdapter,
        sample_documents: list[Document],
    ) -> None:
        """Test that mismatched documents and embeddings raises ValueError."""
        embeddings = [[1.0, 0.0]]  # Only 1 embedding for 3 documents

        with pytest.raises(ValueError, match="must have the same length"):
            adapter.add_documents(sample_documents, embeddings, "test-book")

    def test_similarity_search(
        self,
        adapter: ChromaAdapter,
        sample_documents: list[Document],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test similarity search returns relevant documents."""
        book_id = "test-book-2"
        adapter.add_documents(sample_documents, sample_embeddings, book_id)

        # Search for something similar to first document
        query_embedding = [0.9, 0.1, 0.0, 0.0, 0.0]  # Close to first embedding
        results = adapter.similarity_search(query_embedding, k=2)

        assert len(results) == 2
        # First result should be most similar (Python doc)
        assert "Python" in results[0].page_content

    def test_similarity_search_with_book_filter(
        self,
        adapter: ChromaAdapter,
        sample_documents: list[Document],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test similarity search filtered by book_id."""
        # Add documents to two different books
        adapter.add_documents(sample_documents, sample_embeddings, "book-a")
        adapter.add_documents(sample_documents, sample_embeddings, "book-b")

        query_embedding = [1.0, 0.0, 0.0, 0.0, 0.0]

        # Search within specific book
        results = adapter.similarity_search(query_embedding, k=10, book_id="book-a")
        assert len(results) == len(sample_documents)

        # All results should have book_id metadata
        for doc in results:
            assert doc.metadata.get("book_id") == "book-a"

    def test_list_books(
        self,
        adapter: ChromaAdapter,
        sample_documents: list[Document],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test listing all indexed books."""
        # Add documents to multiple books
        adapter.add_documents(sample_documents, sample_embeddings, "book-1")
        adapter.add_documents([sample_documents[0]], [sample_embeddings[0]], "book-2")

        books = adapter.list_books()

        assert len(books) == 2
        book_ids = {b["id"] for b in books}
        assert book_ids == {"book-1", "book-2"}

        # Check counts
        book1 = next(b for b in books if b["id"] == "book-1")
        assert book1["chunk_count"] == 3

        book2 = next(b for b in books if b["id"] == "book-2")
        assert book2["chunk_count"] == 1

    def test_list_books_empty(self, adapter: ChromaAdapter) -> None:
        """Test listing books when none are indexed."""
        books = adapter.list_books()
        assert books == []

    def test_delete_book_success(
        self,
        adapter: ChromaAdapter,
        sample_documents: list[Document],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test successfully deleting a book."""
        book_id = "book-to-delete"
        adapter.add_documents(sample_documents, sample_embeddings, book_id)

        # Verify book exists
        assert adapter.get_book_chunk_count(book_id) == 3

        # Delete book
        result = adapter.delete_book(book_id)
        assert result is True

        # Verify book is gone
        assert adapter.get_book_chunk_count(book_id) == 0

    def test_delete_book_not_found(self, adapter: ChromaAdapter) -> None:
        """Test deleting a non-existent book returns False."""
        result = adapter.delete_book("non-existent-book")
        assert result is False

    def test_get_book_chunk_count(
        self,
        adapter: ChromaAdapter,
        sample_documents: list[Document],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test getting chunk count for a book."""
        book_id = "count-test-book"
        adapter.add_documents(sample_documents, sample_embeddings, book_id)

        count = adapter.get_book_chunk_count(book_id)
        assert count == len(sample_documents)

    def test_get_book_chunk_count_not_found(self, adapter: ChromaAdapter) -> None:
        """Test getting chunk count for non-existent book returns 0."""
        count = adapter.get_book_chunk_count("unknown-book")
        assert count == 0

    def test_clear(
        self,
        adapter: ChromaAdapter,
        sample_documents: list[Document],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test clearing all documents from the vector store."""
        adapter.add_documents(sample_documents, sample_embeddings, "book-1")
        adapter.add_documents(sample_documents, sample_embeddings, "book-2")

        # Clear all
        adapter.clear()

        # Verify all books are gone
        assert adapter.get_book_chunk_count("book-1") == 0
        assert adapter.get_book_chunk_count("book-2") == 0
        assert adapter.list_books() == []

    def test_search_returns_scores(
        self,
        adapter: ChromaAdapter,
        sample_documents: list[Document],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test that search results include relevance scores."""
        book_id = "score-test-book"
        adapter.add_documents(sample_documents, sample_embeddings, book_id)

        query_embedding = [1.0, 0.0, 0.0, 0.0, 0.0]
        results = adapter.similarity_search(query_embedding, k=1)

        assert len(results) == 1
        assert "score" in results[0].metadata
        assert isinstance(results[0].metadata["score"], float)
        assert 0.0 <= results[0].metadata["score"] <= 1.0
