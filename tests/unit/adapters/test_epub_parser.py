"""Tests for EPUB parser adapter."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from ebook_rag_explorer.adapters.parsers.epub_parser import EpubParser


class TestEpubParser:
    """Test cases for EpubParser."""

    @pytest.fixture
    def parser(self) -> EpubParser:
        """Create an EPUB parser instance."""
        return EpubParser()

    def test_format_name(self, parser: EpubParser) -> None:
        """Test that format_name returns 'EPUB'."""
        assert parser.format_name == "EPUB"

    def test_supports_epub_extension(self, parser: EpubParser) -> None:
        """Test that parser supports .epub extension."""
        assert parser.supports(".epub") is True
        assert parser.supports(".EPUB") is True

    def test_does_not_support_other_extensions(self, parser: EpubParser) -> None:
        """Test that parser does not support non-EPUB extensions."""
        assert parser.supports(".pdf") is False
        assert parser.supports(".txt") is False
        assert parser.supports(".mobi") is False

    def test_parse_nonexistent_file_raises_error(self, parser: EpubParser, tmp_path: Path) -> None:
        """Test that parsing a non-existent file raises FileNotFoundError."""
        nonexistent_file = tmp_path / "nonexistent.epub"

        with pytest.raises(FileNotFoundError, match="File not found"):
            parser.parse(nonexistent_file)

    def test_parse_unsupported_extension_raises_error(
        self, parser: EpubParser, tmp_path: Path
    ) -> None:
        """Test that parsing a file with unsupported extension raises ValueError."""
        wrong_file = tmp_path / "document.pdf"
        wrong_file.write_text("This is not an EPUB")

        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse(wrong_file)

    def test_html_to_text(self, parser: EpubParser) -> None:
        """Test HTML to text conversion."""
        html = b"<html><body><p>Hello  World</p></body></html>"
        text = parser._html_to_text(html)
        assert "Hello" in text
        assert "World" in text

    def test_html_to_text_removes_scripts(self, parser: EpubParser) -> None:
        """Test that script tags are removed."""
        html = b"<html><body><script>alert('test')</script><p>Content</p></body></html>"
        text = parser._html_to_text(html)
        assert "alert" not in text
        assert "Content" in text

    def test_extract_chapter_title_h1(self, parser: EpubParser) -> None:
        """Test extracting chapter title from h1 tag."""
        html = b"<html><body><h1>Chapter Title</h1><p>Content</p></body></html>"
        title = parser._extract_chapter_title(html)
        assert title == "Chapter Title"

    def test_extract_chapter_title_h2_fallback(self, parser: EpubParser) -> None:
        """Test extracting chapter title from h2 when no h1."""
        html = b"<html><body><h2>Section Title</h2><p>Content</p></body></html>"
        title = parser._extract_chapter_title(html)
        assert title == "Section Title"

    def test_extract_chapter_title_none(self, parser: EpubParser) -> None:
        """Test extracting chapter title returns None when no title found."""
        html = b"<html><body><p>Content only</p></body></html>"
        title = parser._extract_chapter_title(html)
        assert title is None

    @patch("ebook_rag_explorer.adapters.parsers.epub_parser.epub.read_epub")
    def test_parse_success(self, mock_read_epub: MagicMock, parser: EpubParser, tmp_path: Path) -> None:
        """Test successful EPUB parsing."""
        # Create mock book
        mock_book = MagicMock()
        mock_book.get_metadata.side_effect = [
            [("Test Book", {})],  # title
            [("Test Author", {})],  # author
        ]

        # Create mock document items
        mock_item1 = MagicMock()
        mock_item1.get_type.return_value = 9  # ITEM_DOCUMENT
        mock_item1.get_id.return_value = "chapter1"
        mock_item1.get_content.return_value = b"<html><body><h1>Chapter 1</h1><p>Content 1</p></body></html>"

        mock_item2 = MagicMock()
        mock_item2.get_type.return_value = 9  # ITEM_DOCUMENT
        mock_item2.get_id.return_value = "chapter2"
        mock_item2.get_content.return_value = b"<html><body><h1>Chapter 2</h1><p>Content 2</p></body></html>"

        # Mock non-document item (should be skipped)
        mock_item3 = MagicMock()
        mock_item3.get_type.return_value = 1  # Not ITEM_DOCUMENT

        mock_book.get_items.return_value = [mock_item1, mock_item2, mock_item3]
        mock_read_epub.return_value = mock_book

        # Create dummy EPUB file
        epub_file = tmp_path / "test.epub"
        epub_file.write_text("dummy")

        # Parse
        documents = parser.parse(epub_file)

        # Assertions
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        assert documents[0].metadata["chapter"] == 1
        assert documents[0].metadata["chapter_title"] == "Chapter 1"
        assert documents[0].metadata["title"] == "Test Book"
        assert documents[0].metadata["author"] == "Test Author"
        assert documents[1].metadata["chapter"] == 2
        assert documents[1].metadata["chapter_title"] == "Chapter 2"

    @patch("ebook_rag_explorer.adapters.parsers.epub_parser.epub.read_epub")
    def test_parse_skips_empty_chapters(
        self, mock_read_epub: MagicMock, parser: EpubParser, tmp_path: Path
    ) -> None:
        """Test that empty chapters are skipped."""
        mock_book = MagicMock()
        mock_book.get_metadata.side_effect = [[], []]  # No metadata

        mock_item1 = MagicMock()
        mock_item1.get_type.return_value = 9
        mock_item1.get_id.return_value = "empty"
        mock_item1.get_content.return_value = b"<html><body>   </body></html>"  # Empty

        mock_item2 = MagicMock()
        mock_item2.get_type.return_value = 9
        mock_item2.get_id.return_value = "content"
        mock_item2.get_content.return_value = b"<html><body><p>Actual content</p></body></html>"

        mock_book.get_items.return_value = [mock_item1, mock_item2]
        mock_read_epub.return_value = mock_book

        epub_file = tmp_path / "test.epub"
        epub_file.write_text("dummy")

        documents = parser.parse(epub_file)

        assert len(documents) == 1
        assert "Actual content" in documents[0].page_content

    @patch("ebook_rag_explorer.adapters.parsers.epub_parser.epub.read_epub")
    def test_parse_handles_epub_error(
        self, mock_read_epub: MagicMock, parser: EpubParser, tmp_path: Path
    ) -> None:
        """Test that epub errors are converted to ValueError."""
        mock_read_epub.side_effect = Exception("Corrupted EPUB")

        epub_file = tmp_path / "test.epub"
        epub_file.write_text("dummy")

        with pytest.raises(ValueError, match="Failed to parse EPUB"):
            parser.parse(epub_file)
