"""Parser port - abstract interface for ebook parsers."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from langchain_core.documents import Document


@runtime_checkable
class EbookParser(Protocol):
    """Protocol for ebook parsers.

    Implementations should parse PDF and EPUB files into LangChain Document objects.
    """

    def parse(self, file_path: Path) -> list[Document]:
        """Parse an ebook file into a list of Documents.

        Args:
            file_path: Path to the ebook file.

        Returns:
            List of Document objects with content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported or parsing fails.
        """
        ...

    def supports(self, file_extension: str) -> bool:
        """Check if this parser supports the given file extension.

        Args:
            file_extension: The file extension including the dot (e.g., ".pdf", ".epub").

        Returns:
            True if this parser can handle the file format.
        """
        ...

    @property
    def format_name(self) -> str:
        """Return the human-readable format name (e.g., "PDF", "EPUB").

        Returns:
            The format name.
        """
        ...
