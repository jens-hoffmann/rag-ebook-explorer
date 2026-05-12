"""API routes for collections management."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status

from ebook_rag_explorer.api.dependencies import get_indexing_service, get_retrieval_service
from ebook_rag_explorer.models import CollectionInfo, ErrorResponse
from ebook_rag_explorer.services.indexing_service import IndexingService
from ebook_rag_explorer.services.retrieval_service import RetrievalService

router = APIRouter()


@router.get(
    "/collections",
    response_model=list[CollectionInfo],
    summary="List all collections",
    description="Get a list of all collections with book and chunk counts.",
)
async def list_collections(
    indexing_service: Annotated[IndexingService, Depends(get_indexing_service)],
) -> list[CollectionInfo]:
    """List all collections.

    Args:
        indexing_service: The indexing service dependency.

    Returns:
        List of CollectionInfo objects.
    """
    collections_data = indexing_service.list_collections()

    return [
        CollectionInfo(
            id=coll["id"],
            name=coll["name"],
            book_count=coll.get("book_count", 0),
            chunk_count=coll.get("chunk_count", 0),
        )
        for coll in collections_data
    ]


@router.delete(
    "/collections/{collection_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
    },
    summary="Delete a collection",
    description="Remove all books and chunks belonging to a specific collection.",
)
async def delete_collection(
    collection_id: Annotated[str, Path(description="The ID of the collection to delete")],
    indexing_service: Annotated[IndexingService, Depends(get_indexing_service)],
) -> None:
    """Delete a collection from the index.

    This will remove all books and their chunks that belong to the specified
    collection.

    Args:
        collection_id: The unique identifier of the collection to delete.
        indexing_service: The indexing service dependency.

    Raises:
        HTTPException: If the collection is not found.
    """
    deleted = indexing_service.delete_collection(collection_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection with ID '{collection_id}' not found",
        )
