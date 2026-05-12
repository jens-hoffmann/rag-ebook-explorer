"""ChromaDB adapter implementation."""

import re
import uuid
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document

from ebook_rag_explorer.ports.vectorstore_port import VectorStore


def normalize_collection_id(name: str) -> str:
    """Normalize a collection name to a valid ID.

    Args:
        name: The collection name to normalize.

    Returns:
        Normalized collection ID (lowercase, stripped, special chars removed).
    """
    # Strip whitespace, lowercase, replace non-alphanumeric with underscore
    normalized = name.strip().lower()
    normalized = re.sub(r"[^a-z0-9_\-]", "_", normalized)
    # Remove consecutive underscores
    normalized = re.sub(r"_+", "_", normalized)
    # Strip leading/trailing underscores
    normalized = normalized.strip("_")
    return normalized or "default"


class ChromaAdapter(VectorStore):
    """ChromaDB implementation of the VectorStore port."""

    COLLECTION_NAME = "ebook_documents"

    def __init__(self, persist_dir: Path | str) -> None:
        """Initialize the ChromaDB adapter.

        Args:
            persist_dir: Directory for ChromaDB persistence.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        book_id: str,
        collection_id: str | None = None,
    ) -> None:
        """Add documents with their embeddings to the vector store.

        Args:
            documents: List of Document objects to store.
            embeddings: List of embedding vectors corresponding to documents.
            book_id: Unique identifier for the book these documents belong to.
            collection_id: Optional collection to organize the book into.

        Raises:
            ValueError: If documents and embeddings have different lengths.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Documents ({len(documents)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )

        if not documents:
            return

        # Normalize collection_id
        normalized_collection_id = normalize_collection_id(collection_id) if collection_id else None

        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        embedding_vectors = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            chunk_id = f"{book_id}_{i}_{uuid.uuid4().hex[:8]}"
            ids.append(chunk_id)
            texts.append(doc.page_content)
            embedding_vectors.append(embedding)

            # Merge document metadata with book_id and collection_id
            metadata: dict[str, Any] = {
                **doc.metadata,
                "book_id": book_id,
                "chunk_index": i,
            }
            if normalized_collection_id:
                metadata["collection_id"] = normalized_collection_id
            metadatas.append(metadata)

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embedding_vectors,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int,
        book_id: str | None = None,
        collection_id: str | None = None,
    ) -> list[Document]:
        """Search for documents similar to the query embedding.

        Args:
            query_embedding: The query vector to search for.
            k: Number of results to return.
            book_id: Optional filter to search within a specific book.
            collection_id: Optional filter to search within a specific collection.

        Returns:
            List of similar Document objects, ordered by relevance.
        """
        # Build where filter
        where_filter: dict[str, Any] | None = None
        conditions = []

        if book_id:
            conditions.append({"book_id": book_id})

        if collection_id:
            normalized = normalize_collection_id(collection_id)
            conditions.append({"collection_id": normalized})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            # Use $and operator for multiple conditions
            where_filter = {"$and": conditions}

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Convert results to Document objects
        documents = []
        if results["documents"] and results["documents"][0]:
            for doc_text, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Add score to metadata (convert distance to similarity)
                metadata_with_score = {
                    **metadata,
                    "score": 1.0 - distance,  # Cosine distance to similarity
                }
                documents.append(
                    Document(
                        page_content=doc_text,
                        metadata=metadata_with_score,
                    )
                )

        return documents

    def delete_book(self, book_id: str) -> bool:
        """Delete all documents belonging to a specific book.

        Args:
            book_id: Unique identifier for the book to delete.

        Returns:
            True if the book was found and deleted, False otherwise.
        """
        # Check if book exists
        count = self.get_book_chunk_count(book_id)
        if count == 0:
            return False

        # Delete by metadata filter
        self.collection.delete(where={"book_id": book_id})
        return True

    def list_books(self) -> list[dict]:
        """List all indexed books with their metadata.

        Returns:
            List of dictionaries containing book info.
        """
        # Get all unique book_ids from metadata
        result = self.collection.get(include=["metadatas"])

        if not result["metadatas"]:
            return []

        # Aggregate by book_id
        books: dict[str, dict] = {}
        for metadata in result["metadatas"]:
            book_id = metadata.get("book_id", "unknown")

            if book_id not in books:
                books[book_id] = {
                    "id": book_id,
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "format": metadata.get("format", ""),
                    "collection_id": metadata.get("collection_id"),
                    "chunk_count": 0,
                }

            books[book_id]["chunk_count"] += 1

        return list(books.values())

    def get_book_chunk_count(self, book_id: str) -> int:
        """Get the number of chunks indexed for a specific book.

        Args:
            book_id: Unique identifier for the book.

        Returns:
            Number of chunks for the book, or 0 if not found.
        """
        result = self.collection.get(
            where={"book_id": book_id},
            include=[],
        )
        return len(result["ids"]) if result["ids"] else 0

    def list_collections(self) -> list[dict]:
        """List all collections with their metadata.

        Returns:
            List of dictionaries containing collection info.
        """
        result = self.collection.get(include=["metadatas"])

        if not result["metadatas"]:
            return []

        # Aggregate by collection_id
        collections: dict[str, dict] = {}
        for metadata in result["metadatas"]:
            collection_id = metadata.get("collection_id")
            if not collection_id:
                continue  # Skip documents without collection

            if collection_id not in collections:
                collections[collection_id] = {
                    "id": collection_id,
                    "name": collection_id,  # Display name is the ID for now
                    "book_count": 0,
                    "chunk_count": 0,
                    "books": set(),
                }

            collections[collection_id]["chunk_count"] += 1
            collections[collection_id]["books"].add(metadata.get("book_id", "unknown"))

        # Convert sets to counts and prepare final list
        result_list = []
        for coll_data in collections.values():
            result_list.append({
                "id": coll_data["id"],
                "name": coll_data["name"],
                "book_count": len(coll_data["books"]),
                "chunk_count": coll_data["chunk_count"],
            })

        # Sort by name for consistent ordering
        result_list.sort(key=lambda x: x["name"])
        return result_list

    def delete_collection(self, collection_id: str) -> bool:
        """Delete all documents belonging to a specific collection.

        Args:
            collection_id: Collection identifier to delete.

        Returns:
            True if the collection was found and deleted, False otherwise.
        """
        normalized = normalize_collection_id(collection_id)

        # Check if collection exists
        result = self.collection.get(
            where={"collection_id": normalized},
            include=[],
        )

        if not result["ids"]:
            return False

        # Delete by collection_id filter
        self.collection.delete(where={"collection_id": normalized})
        return True

    def clear(self) -> None:
        """Clear all documents from the vector store."""
        # Delete and recreate collection
        self.client.delete_collection(name=self.COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
