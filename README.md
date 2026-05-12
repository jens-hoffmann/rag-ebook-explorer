# Ebook RAG Explorer

A RAG application with FastAPI backend that indexes EPUB/PDF ebooks via sentence-transformers into **PostgreSQL with pgvector**, retrieves with **hybrid search (vector similarity + full-text search)**, and generates answers using a configurable LLM.

## Features

- **Document Parsing**: Support for PDF (PyMuPDF) and EPUB (ebooklib) formats
- **Hybrid Search**: Combines vector similarity (pgvector) with full-text search (PostgreSQL tsquery)
- **Reranking**: CrossEncoder for improved retrieval quality
- **LLM Integration**: Configurable support for OpenAI, Azure OpenAI, LM Studio
- **Hexagonal Architecture**: Clean separation with Ports, Adapters, and Services
- **Collections**: Organize books into user-defined collections
- **Container-Native**: Designed for Podman/Docker deployment

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Hybrid Search (Postgres)                        │
│  ┌─────────────────────┐    ┌──────────────────────────────────────┐   │
│  │   Vector Search     │    │      Full-Text Search                │   │
│  │   (pgvector)        │    │      (tsquery/tsvector)              │   │
│  │   embedding ⇔ query │    │  to_tsvector(content) @@ query       │   │
│  └──────────┬──────────┘    └──────────────┬───────────────────────┘   │
│             │                              │                            │
│             └──────────────┬───────────────┘                            │
│                            ▼                                            │
│                   ┌─────────────────┐                                   │
│                   │  Reciprocal Rank│  RRF Score = Σ(1/(k + rank))     │
│                   │  Fusion (RRF)   │  k=60                            │
│                   └────────┬────────┘                                   │
│                            ▼                                            │
│                   ┌─────────────────┐                                   │
│                   │ Combined Results│                                   │
│                   └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start with Podman

### Prerequisites

- [Podman](https://podman.io/) and podman-compose
- Or Docker and docker-compose

### 1. Clone and Configure

```bash
git clone https://github.com/user/ebook-rag-explorer.git
cd ebook-rag-explorer

# Copy and edit environment file
cp .env.example .env
# Edit .env with your settings (at minimum LLM_API_KEY)
```

### 2. Start the Stack

```bash
# Start PostgreSQL and Backend
podman-compose up -d

# View logs
podman-compose logs -f backend

# Check health
curl http://localhost:8000/health
```

### 3. Use the API

```bash
# Index a book
curl -X POST "http://localhost:8000/api/index" \
  -F "file=@mybook.pdf" \
  -F "collection_id=Technical Books"

# Search with hybrid retrieval
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "collection_id": "Technical Books"}'

# List collections
curl "http://localhost:8000/api/collections"

# List books
curl "http://localhost:8000/api/books"

# Delete a book
curl -X DELETE "http://localhost:8000/api/books/{book_id}"
```

### 4. Stop the Stack

```bash
podman-compose down

# To also remove volumes (database data):
podman-compose down -v
```

## Development Setup

### Local Development (without containers)

```bash
# Install dependencies with uv
uv sync

# Start PostgreSQL locally (or use compose for just the database)
podman-compose up -d postgres

# Run the API server
uv run uvicorn ebook_rag_explorer.api.app:create_app --reload

# Or use tox
tox -e serve
```

### Running Tests

```bash
# All tests
uv run pytest tests -v

# Unit tests only (no database required)
uv run pytest tests/unit -v

# Integration tests (requires PostgreSQL)
# Option 1: Use running database
uv run pytest tests/integration -v

# Option 2: Use testcontainers (automatic PostgreSQL)
tox -e integration

# Or with tox
tox -e unit
tox -e integration
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb` | Database connection |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence transformer model |
| `EMBEDDING_DIMENSION` | `384` | Vector dimension (match your model) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking |
| `LLM_PROVIDER` | `openai` | LLM provider (openai, lmstudio, azure) |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `LLM_API_KEY` | - | API key for LLM |
| `RETRIEVAL_TOP_K` | `20` | Vector search results |
| `TEXT_SEARCH_TOP_K` | `20` | Full-text search results |
| `HYBRID_TOP_K` | `40` | Combined results before reranking |
| `RRF_K` | `60` | RRF fusion parameter |
| `RERANK_TOP_N` | `5` | Final results after reranking |
| `CHUNK_SIZE` | `1000` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |

## API Endpoints

- `POST /api/index` - Upload and index PDF/EPUB files
- `POST /api/search` - Hybrid search with RAG
- `GET /api/books` - List all indexed books
- `DELETE /api/books/{id}` - Delete a book
- `GET /api/collections` - List collections
- `DELETE /api/collections/{id}` - Delete a collection
- `GET /health` - Health check

## Database Schema

The application uses PostgreSQL with pgvector extension:

```sql
-- Documents table with hybrid search
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) UNIQUE NOT NULL,
    book_id VARCHAR(255) REFERENCES books(id),
    collection_id VARCHAR(255),
    content TEXT NOT NULL,
    embedding VECTOR(384),  -- For all-MiniLM-L6-v2
    search_vector tsvector, -- For full-text search
    -- ... metadata columns
);

-- Indexes for performance
CREATE INDEX idx_documents_embedding ON documents 
    USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_documents_search ON documents 
    USING GIN (search_vector);
```

## Project Structure

```
ebook-rag-explorer/
├── compose.yaml           # Podman Compose stack
├── Containerfile          # Backend container image
├── init-db/
│   └── 01-init.sql      # Database schema
├── src/
│   └── ebook_rag_explorer/
│       ├── adapters/     # Concrete implementations
│       │   ├── parsers/  # PDF/EPUB parsers
│       │   ├── vectorstore/
│       │   │   └── postgres_adapter.py  # PostgreSQL + pgvector
│       │   ├── retrieval/
│       │   │   ├── postgres_retriever.py # Hybrid search
│       │   │   └── cross_encoder_reranker.py
│       │   └── llm/
│       ├── services/     # Business logic
│       ├── ports/        # Abstract interfaces
│       └── api/          # FastAPI application
└── tests/
```

## License

MIT License
