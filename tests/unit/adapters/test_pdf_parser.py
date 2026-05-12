"""Tests for PDF parser adapter."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from ebook_rag_explorer.adapters.parsers.pdf_parser import PdfParser


class TestPdfParser:
    """Test cases for PdfParser."""

    @pytest.fixture
    def parser(self) -> PdfParser:
        """Create a PDF parser instance."""
        return PdfParser()

    def test_format_name(self, parser: PdfParser) -> None:
        """Test that format_name returns 'PDF'."""
        assert parser.format_name == "PDF"

    def test_supports_pdf_extension(self, parser: PdfParser) -> None:
        """Test that parser supports .pdf extension."""
        assert parser.supports(".pdf") is True
        assert parser.supports(".PDF") is True

    def test_does_not_support_other_extensions(self, parser: PdfParser) -> None:
        """Test that parser does not support non-PDF extensions."""
        assert parser.supports(".epub") is False
        assert parser.supports(".txt") is False
        assert parser.supports(".docx") is False

    def test_parse_nonexistent_file_raises_error(self, parser: PdfParser, tmp_path: Path) -> None:
        """Test that parsing a non-existent file raises FileNotFoundError."""
        nonexistent_file = tmp_path / "nonexistent.pdf"

        with pytest.raises(FileNotFoundError, match="File not found"):
            parser.parse(nonexistent_file)

    def test_parse_unsupported_extension_raises_error(
        self, parser: PdfParser, tmp_path: Path
    ) -> None:
        """Test that parsing a file with unsupported extension raises ValueError."""
        # Create a dummy file with wrong extension
        wrong_file = tmp_path / "document.txt"
        wrong_file.write_text("This is not a PDF")

        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse(wrong_file)

    @patch("ebook_rag_explorer.adapters.parsers.pdf_parser.fitz.open")
    def test_parse_success(self, mock_fitz_open: MagicMock, parser: PdfParser, tmp_path: Path) -> None:
        """Test successful PDF parsing."""
        # Create a mock PDF document
        mock_doc = MagicMock()
        mock_doc.metadata = {"title": "Test Book", "author": "Test Author"}
        mock_doc.__len__ = MagicMock(return_value=2)

        # Mock pages
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2 content"

        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_doc.__getitem__ = MagicMock(side_effect=[mock_page1, mock_page2])

        mock_fitz_open.return_value = mock_doc

        # Create a dummy PDF file path
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("dummy")  # Content doesn't matter due to mocking

        # Parse
        documents = parser.parse(pdf_file)

        # Assertions
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        assert documents[0].page_content == "Page 1 content"
        assert documents[0].metadata["page"] == 1
        assert documents[0].metadata["title"] == "Test Book"
        assert documents[0].metadata["author"] == "Test Author"
        assert documents[1].page_content == "Page 2 content"
        assert documents[1].metadata["page"] == 2

    @patch("ebook_rag_explorer.adapters.parsers.pdf_parser.fitz.open")
    def test_parse_skips_empty_pages(
        self, mock_fitz_open: MagicMock, parser: PdfParser, tmp_path: Path
    ) -> None:
        """Test that empty pages are skipped."""
        # Create a mock PDF document with one empty page
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_doc.__len__ = MagicMock(return_value=2)

        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "   "  # Empty/whitespace only
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Actual content"

        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_doc.__getitem__ = MagicMock(side_effect=[mock_page1, mock_page2])

        mock_fitz_open.return_value = mock_doc

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("dummy")

        documents = parser.parse(pdf_file)

        # Only the non-empty page should be included
        assert len(documents) == 1
        assert documents[0].page_content == "Actual content"

    @patch("ebook_rag_explorer.adapters.parsers.pdf_parser.fitz.open")
    def test_parse_handles_fitz_error(
        self, mock_fitz_open: MagicMock, parser: PdfParser, tmp_path: Path
    ) -> None:
        """Test that fitz errors are converted to ValueError."""
        mock_fitz_open.side_effect = Exception("Corrupted PDF")

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("dummy")

        with pytest.raises(ValueError, match="Failed to parse PDF"):
            parser.parse(pdf_file)
