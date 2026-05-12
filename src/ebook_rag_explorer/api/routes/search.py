"""API routes for search."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ebook_rag_explorer.api.dependencies import get_retrieval_service
from ebook_rag_explorer.models import ErrorResponse, SearchRequest, SearchResponse
from ebook_rag_explorer.services.retrieval_service import RetrievalService

router = APIRouter()


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Search indexed documents",
    description="Search the indexed documents using semantic search with RAG.",
)
async def search_documents(
    request: SearchRequest,
    retrieval_service: Annotated[
        RetrievalService,
        Depends(get_retrieval_service),
    ],
) -> SearchResponse:
    """Search indexed documents.

    This endpoint performs semantic search across indexed books,
    reranks results using a cross-encoder, and generates an answer
    using the configured LLM.

    Args:
        request: The search request with query and optional parameters.
        retrieval_service: The retrieval service dependency.

    Returns:
        SearchResponse with answer and sources.

    Raises:
        HTTPException: If search fails.
    """
    try:
        response = retrieval_service.search(
            query=request.query,
            top_k=request.top_k,
        )
        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e}",
        ) from e
