"""EPUB parser adapter using ebooklib."""

import re
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from langchain_core.documents import Document

from ebook_rag_explorer.ports.parser_port import EbookParser


class EpubParser:
    """Parser for EPUB files using ebooklib."""

    SUPPORTED_EXTENSIONS = {".epub"}

    @property
    def format_name(self) -> str:
        """Return the format name."""
        return "EPUB"

    def supports(self, file_extension: str) -> bool:
        """Check if this parser supports the given file extension.

        Args:
            file_extension: The file extension including the dot.

        Returns:
            True if the extension is supported.
        """
        return file_extension.lower() in self.SUPPORTED_EXTENSIONS

    def _html_to_text(self, html_content: bytes) -> str:
        """Convert HTML content to plain text.

        Args:
            html_content: Raw HTML bytes.

        Returns:
            Cleaned plain text.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text

    def _extract_chapter_title(self, html_content: bytes) -> str | None:
        """Try to extract chapter title from HTML.

        Args:
            html_content: Raw HTML bytes.

        Returns:
            Chapter title if found, None otherwise.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Try to find h1, h2, or title
        for tag in ["h1", "h2", "title"]:
            element = soup.find(tag)
            if element:
                return element.get_text().strip()

        return None

    def parse(self, file_path: Path) -> list[Document]:
        """Parse an EPUB file into a list of Documents.

        Args:
            file_path: Path to the EPUB file.

        Returns:
            List of Document objects, one per chapter/document with metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.supports(file_path.suffix):
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        documents = []

        try:
            book = epub.read_epub(file_path)

            # Extract book metadata
            title = book.get_metadata("DC", "title")
            author = book.get_metadata("DC", "creator")

            metadata = {
                "source": str(file_path),
                "format": "epub",
                "title": title[0][0] if title else "",
                "author": author[0][0] if author else "",
            }

            # Process each document item (chapters)
            chapter_num = 0
            for item in book.get_items():
                if item.get_type() == ITEM_DOCUMENT:
                    chapter_num += 1
                    html_content = item.get_content()

                    # Extract text
                    text = self._html_to_text(html_content)

                    if text.strip():  # Only add non-empty chapters
                        chapter_title = self._extract_chapter_title(html_content)

                        chapter_metadata = {
                            **metadata,
                            "chapter": chapter_num,
                            "chapter_title": chapter_title or f"Chapter {chapter_num}",
                            "item_id": item.get_id(),
                        }

                        documents.append(
                            Document(
                                page_content=text.strip(),
                                metadata=chapter_metadata,
                            )
                        )

        except Exception as e:
            raise ValueError(f"Failed to parse EPUB: {e}") from e

        return documents
