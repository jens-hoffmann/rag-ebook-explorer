"""Indexing service for orchestrating the document indexing pipeline."""

import uuid
from pathlib import Path

from langchain_core.documents import Document

from ebook_rag_explorer.adapters.parsers.epub_parser import EpubParser
from ebook_rag_explorer.adapters.parsers.pdf_parser import PdfParser
from ebook_rag_explorer.ports.embedder_port import Embedder
from ebook_rag_explorer.ports.parser_port import EbookParser
from ebook_rag_explorer.ports.vectorstore_port import VectorStore
from ebook_rag_explorer.services.chunking_service import ChunkingService


class IndexingService:
    """Service for indexing ebooks into the vector store."""

    # Registry of available parsers
    PARSERS: list[type[EbookParser]] = [PdfParser, EpubParser]

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        chunking_service: ChunkingService,
    ) -> None:
        """Initialize the indexing service.

        Args:
            vector_store: The vector store to save documents to.
            embedder: The embedder to generate embeddings.
            chunking_service: Service for chunking documents.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunking_service = chunking_service

    def _get_parser(self, file_path: Path) -> EbookParser:
        """Get the appropriate parser for the file format.

        Args:
            file_path: Path to the ebook file.

        Returns:
            An EbookParser instance that supports the file.

        Raises:
            ValueError: If no parser supports the file format.
        """
        extension = file_path.suffix.lower()

        for parser_class in self.PARSERS:
            parser = parser_class()
            if parser.supports(extension):
                return parser

        raise ValueError(f"Unsupported file format: {extension}")

    def index_book(
        self,
        file_path: Path,
        book_id: str | None = None,
        collection_id: str | None = None,
    ) -> dict:
        """Index an ebook file into the vector store.

        This orchestrates the full pipeline: parse → chunk → embed → store.

        Args:
            file_path: Path to the ebook file.
            book_id: Optional unique identifier for the book.
                     If not provided, a UUID will be generated.
            collection_id: Optional collection to organize the book into.

        Returns:
            Dictionary with indexing results:
            {
                "book_id": str,
                "title": str | None,
                "format": str,
                "chunks_indexed": int,
                "collection_id": str | None,
            }

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
            RuntimeError: If indexing fails.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate book_id if not provided
        if book_id is None:
            book_id = f"book_{uuid.uuid4().hex[:12]}"

        # Step 1: Parse
        parser = self._get_parser(file_path)
        documents = parser.parse(file_path)

        if not documents:
            return {
                "book_id": book_id,
                "title": None,
                "format": parser.format_name,
                "chunks_indexed": 0,
                "collection_id": collection_id,
            }

        # Extract title from first document metadata
        title = documents[0].metadata.get("title")

        # Step 2: Chunk
        chunks = self.chunking_service.chunk_documents(documents)

        if not chunks:
            return {
                "book_id": book_id,
                "title": title,
                "format": parser.format_name,
                "chunks_indexed": 0,
                "collection_id": collection_id,
            }

        # Step 3: Embed
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)

        # Step 4: Store (with collection_id)
        self.vector_store.add_documents(chunks, embeddings, book_id, collection_id)

        return {
            "book_id": book_id,
            "title": title,
            "format": parser.format_name,
            "chunks_indexed": len(chunks),
            "collection_id": collection_id,
        }

    def delete_book(self, book_id: str) -> bool:
        """Delete a book from the index.

        Args:
            book_id: Unique identifier for the book.

        Returns:
            True if the book was found and deleted, False otherwise.
        """
        return self.vector_store.delete_book(book_id)

    def list_books(self) -> list[dict]:
        """List all indexed books.

        Returns:
            List of book information dictionaries.
        """
        return self.vector_store.list_books()

    def list_collections(self) -> list[dict]:
        """List all collections.

        Returns:
            List of collection information dictionaries.
        """
        return self.vector_store.list_collections()

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection and all its books.

        Args:
            collection_id: The collection identifier to delete.

        Returns:
            True if the collection was found and deleted, False otherwise.
        """
        return self.vector_store.delete_collection(collection_id)
