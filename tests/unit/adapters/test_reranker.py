"""Tests for CrossEncoder reranker."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from ebook_rag_explorer.adapters.retrieval.cross_encoder_reranker import (
    CrossEncoderReranker,
)


class TestCrossEncoderReranker:
    """Test cases for CrossEncoderReranker."""

    @pytest.fixture
    def reranker(self) -> CrossEncoderReranker:
        """Create a reranker instance with mock model."""
        with patch("ebook_rag_explorer.adapters.retrieval.cross_encoder_reranker.CrossEncoder"):
            return CrossEncoderReranker("cross-encoder/test-model")

    def test_model_name_property(self, reranker: CrossEncoderReranker) -> None:
        """Test that model_name returns the configured model."""
        assert reranker.model_name == "cross-encoder/test-model"

    def test_rerank_empty_list(self, reranker: CrossEncoderReranker) -> None:
        """Test reranking an empty list returns empty."""
        result = reranker.rerank("query", [], top_n=5)
        assert result == []

    def test_rerank_returns_top_n(self, reranker: CrossEncoderReranker) -> None:
        """Test that reranking returns only top_n documents."""
        with patch.object(reranker, "_get_model") as mock_get_model:
            mock_model = MagicMock()
            # Return descending scores so documents are reordered
            mock_model.predict.return_value = [0.9, 0.7, 0.8, 0.6, 0.5]
            mock_get_model.return_value = mock_model

            documents = [
                Document(page_content=f"Doc {i}", metadata={})
                for i in range(5)
            ]

            result = reranker.rerank("query", documents, top_n=3)

            assert len(result) == 3
            # Highest scored document (Doc 0 with score 0.9) should be first
            assert result[0].page_content == "Doc 0"
            assert result[0].metadata["rerank_score"] == 0.9

    def test_rerank_adds_scores_to_metadata(self, reranker: CrossEncoderReranker) -> None:
        """Test that reranking adds rerank_score to document metadata."""
        with patch.object(reranker, "_get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.5, 0.8]
            mock_get_model.return_value = mock_model

            documents = [
                Document(page_content="First", metadata={"page": 1}),
                Document(page_content="Second", metadata={"page": 2}),
            ]

            result = reranker.rerank("query", documents, top_n=1)  # top_n < len(docs) to trigger sorting

            # Scores should be in metadata
            # Documents are reordered by score, so highest score (0.8) is first
            assert "rerank_score" in result[0].metadata
            assert result[0].metadata["rerank_score"] == 0.8  # Higher score first
            assert result[0].metadata["page"] == 2  # Document with score 0.8

    def test_rerank_preserves_original_metadata(self, reranker: CrossEncoderReranker) -> None:
        """Test that reranking preserves original document metadata."""
        with patch.object(reranker, "_get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.5]
            mock_get_model.return_value = mock_model

            documents = [
                Document(
                    page_content="Test",
                    metadata={"page": 5, "title": "Test Doc", "custom": "value"},
                ),
            ]

            result = reranker.rerank("query", documents, top_n=1)

            assert result[0].metadata["page"] == 5
            assert result[0].metadata["title"] == "Test Doc"
            assert result[0].metadata["custom"] == "value"

    def test_rerank_returns_all_if_fewer_than_top_n(
        self, reranker: CrossEncoderReranker
    ) -> None:
        """Test that all documents are returned if fewer than top_n."""
        with patch.object(reranker, "_get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.9, 0.5]
            mock_get_model.return_value = mock_model

            documents = [
                Document(page_content="Doc 1", metadata={}),
                Document(page_content="Doc 2", metadata={}),
            ]

            result = reranker.rerank("query", documents, top_n=10)

            assert len(result) == 2
            # Both should have scores added
            assert "rerank_score" in result[0].metadata
            assert "rerank_score" in result[1].metadata

    def test_rerank_query_document_pairs(self, reranker: CrossEncoderReranker) -> None:
        """Test that reranking creates correct query-document pairs."""
        with patch.object(reranker, "_get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.9, 0.8, 0.7]
            mock_get_model.return_value = mock_model

            documents = [
                Document(page_content="Content A", metadata={}),
                Document(page_content="Content B", metadata={}),
                Document(page_content="Content C", metadata={}),
            ]

            reranker.rerank("search query", documents, top_n=2)

            # Verify predict was called with correct pairs
            expected_pairs = [
                ["search query", "Content A"],
                ["search query", "Content B"],
                ["search query", "Content C"],
            ]
            mock_model.predict.assert_called_once_with(expected_pairs)
