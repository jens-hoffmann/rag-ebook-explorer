"""Chunking service for splitting documents into smaller chunks."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingService:
    """Service for chunking documents using recursive character text splitting."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Initialize the chunking service.

        Args:
            chunk_size: Target size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_document(self, document: Document) -> list[Document]:
        """Split a single document into chunks.

        Args:
            document: The document to chunk.

        Returns:
            List of chunked Document objects with preserved metadata.
        """
        chunks = self._splitter.split_documents([document])

        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata = {
                **chunk.metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split multiple documents into chunks.

        Args:
            documents: List of documents to chunk.

        Returns:
            Flat list of all chunked Document objects.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
