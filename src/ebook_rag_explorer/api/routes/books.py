"""API routes for book management."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from ebook_rag_explorer.api.dependencies import get_indexing_service, get_vector_store
from ebook_rag_explorer.models import BookInfo, ErrorResponse
from ebook_rag_explorer.services.indexing_service import IndexingService
from ebook_rag_explorer.ports.vectorstore_port import VectorStore

router = APIRouter()


@router.get(
    "/books",
    response_model=list[BookInfo],
    summary="List all indexed books",
    description="Get a list of all books currently indexed in the vector store.",
)
async def list_books(
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    collection_id: Annotated[str | None, Query(description="Filter by collection ID")] = None,
) -> list[BookInfo]:
    """List all indexed books.

    Args:
        vector_store: The vector store dependency.
        collection_id: Optional filter to list books in a specific collection.

    Returns:
        List of BookInfo objects.
    """
    books_data = vector_store.list_books()

    # Filter by collection_id if provided
    if collection_id:
        from ebook_rag_explorer.adapters.vectorstore.chroma_adapter import normalize_collection_id
        normalized = normalize_collection_id(collection_id)
        books_data = [b for b in books_data if b.get("collection_id") == normalized]

    return [
        BookInfo(
            id=book["id"],
            title=book.get("title") or None,
            format=book.get("format", "unknown"),
            chunk_count=book.get("chunk_count", 0),
            collection_id=book.get("collection_id"),
            metadata={
                "author": book.get("author", ""),
            },
        )
        for book in books_data
    ]


@router.delete(
    "/books/{book_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Book not found"},
    },
    summary="Delete a book",
    description="Remove all chunks of a book from the vector store.",
)
async def delete_book(
    book_id: Annotated[str, Path(description="The ID of the book to delete")],
    indexing_service: Annotated[
        IndexingService,
        Depends(get_indexing_service),
    ],
) -> None:
    """Delete a book from the index.

    Args:
        book_id: The unique identifier of the book to delete.
        indexing_service: The indexing service dependency.

    Raises:
        HTTPException: If the book is not found.
    """
    deleted = indexing_service.delete_book(book_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book with ID '{book_id}' not found",
        )
