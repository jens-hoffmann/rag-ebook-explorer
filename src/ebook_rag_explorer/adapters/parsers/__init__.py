"""Parser adapters module."""

from ebook_rag_explorer.adapters.parsers.epub_parser import EpubParser
from ebook_rag_explorer.adapters.parsers.pdf_parser import PdfParser

__all__ = ["PdfParser", "EpubParser"]
