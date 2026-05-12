"""PDF parser adapter using PyMuPDF."""

from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document

from ebook_rag_explorer.ports.parser_port import EbookParser


class PdfParser:
    """Parser for PDF files using PyMuPDF."""

    SUPPORTED_EXTENSIONS = {".pdf"}

    @property
    def format_name(self) -> str:
        """Return the format name."""
        return "PDF"

    def supports(self, file_extension: str) -> bool:
        """Check if this parser supports the given file extension.

        Args:
            file_extension: The file extension including the dot.

        Returns:
            True if the extension is supported.
        """
        return file_extension.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, file_path: Path) -> list[Document]:
        """Parse a PDF file into a list of Documents.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of Document objects, one per page with metadata.

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
            with fitz.open(file_path) as doc:
                # Extract document metadata
                metadata = {
                    "source": str(file_path),
                    "format": "pdf",
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "total_pages": len(doc),
                }

                # Extract text from each page
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()

                    if text.strip():  # Only add non-empty pages
                        page_metadata = {
                            **metadata,
                            "page": page_num + 1,  # 1-indexed page numbers
                        }
                        documents.append(
                            Document(
                                page_content=text.strip(),
                                metadata=page_metadata,
                            )
                        )

        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {e}") from e

        return documents
