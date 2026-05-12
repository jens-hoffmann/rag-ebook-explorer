"""API routes for document indexing."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from ebook_rag_explorer.api.dependencies import get_indexing_service
from ebook_rag_explorer.models import ErrorResponse, IndexResponse
from ebook_rag_explorer.services.indexing_service import IndexingService

router = APIRouter()


@router.post(
    "/index",
    response_model=IndexResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Upload and index a document",
    description="Upload a PDF or EPUB file to be indexed for semantic search.",
)
async def index_document(
    file: Annotated[UploadFile, File(description="PDF or EPUB file to index")],
    book_id: Annotated[str | None, Form(description="Optional custom book ID")] = None,
    collection_id: Annotated[str | None, Form(description="Optional collection to organize the book into")] = None,
    indexing_service: IndexingService = Depends(get_indexing_service),
) -> IndexResponse:
    """Upload and index a document.

    This endpoint accepts PDF or EPUB files, parses them, chunks the content,
    generates embeddings, and stores them in the vector database.

    Args:
        file: The document file to index.
        book_id: Optional custom identifier for the book.
        collection_id: Optional collection to organize the book into.
        indexing_service: The indexing service dependency.

    Returns:
        IndexResponse with indexing results.

    Raises:
        HTTPException: If file format is unsupported or indexing fails.
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in {".pdf", ".epub"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {file_extension}. Use PDF or EPUB.",
        )

    # Save uploaded file temporarily
    temp_path = Path(f"/tmp/{file.filename}")
    try:
        content = await file.read()
        temp_path.write_bytes(content)

        # Index the book
        result = indexing_service.index_book(
            file_path=temp_path,
            book_id=book_id,
            collection_id=collection_id,
        )

        return IndexResponse(
            book_id=result["book_id"],
            title=result.get("title"),
            chunks_indexed=result["chunks_indexed"],
            format=result["format"],
            message="Book indexed successfully",
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File error: {e}",
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file: {e}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {e}",
        ) from e
    finally:
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()
