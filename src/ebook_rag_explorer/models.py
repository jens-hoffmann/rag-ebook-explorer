"""Shared Pydantic models for request/response data transfer objects."""

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """A source document returned in search results."""

    content: str = Field(description="The document content/chunk text")
    score: float = Field(description="Relevance score from retrieval")
    metadata: dict = Field(
        default_factory=dict,
        description="Document metadata (page, chapter, source file, etc.)",
    )


class SearchRequest(BaseModel):
    """Request model for the search endpoint."""

    query: str = Field(..., min_length=1, description="The search query")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Override default number of documents to retrieve",
    )


class SearchResponse(BaseModel):
    """Response model for the search endpoint."""

    query: str = Field(description="The original search query")
    answer: str = Field(description="The generated answer from the LLM")
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Source documents used to generate the answer",
    )
    retrieved_count: int = Field(
        default=0,
        description="Number of documents initially retrieved",
    )
    reranked_count: int = Field(
        default=0,
        description="Number of documents after reranking",
    )


class BookInfo(BaseModel):
    """Information about an indexed book."""

    id: str = Field(description="Unique identifier for the book")
    title: str | None = Field(
        default=None,
        description="Book title from metadata",
    )
    format: str = Field(
        description="File format (pdf, epub, etc.)",
        examples=["pdf", "epub"],
    )
    chunk_count: int = Field(
        default=0,
        description="Number of chunks indexed for this book",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional book metadata (author, pages, etc.)",
    )


class IndexResponse(BaseModel):
    """Response model for the index endpoint."""

    book_id: str = Field(description="Unique identifier for the indexed book")
    title: str | None = Field(
        default=None,
        description="Book title extracted from metadata",
    )
    chunks_indexed: int = Field(
        description="Number of chunks successfully indexed",
    )
    format: str = Field(description="Detected file format")
    message: str = Field(
        default="Book indexed successfully",
        description="Status message",
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(description="Error type/code")
    message: str = Field(description="Human-readable error message")
    details: dict | None = Field(
        default=None,
        description="Additional error details",
    )
