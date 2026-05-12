# Ebook RAG Explorer

A RAG application with FastAPI backend that indexes EPUB/PDF ebooks via sentence-transformers + langchain into ChromaDB, retrieves with semantic search + CrossEncoder reranking, and generates answers using a configurable LLM.

## Features

- **Document Parsing**: Support for PDF (PyMuPDF) and EPUB (ebooklib) formats
- **Semantic Search**: Vector embeddings using sentence-transformers
- **Reranking**: CrossEncoder for improved retrieval quality
- **LLM Integration**: Configurable support for OpenAI, Azure OpenAI, LM Studio
- **Hexagonal Architecture**: Clean separation with Ports, Adapters, and Services
- **Testing**: Comprehensive unit and integration tests with testcontainers (Podman)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              API Layer                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                       │
│  │ /api/index  │  │/api/search  │  │ /api/books  │                       │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                       │
└─────────┼────────────────┼────────────────┼─────────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Services (Domain Logic)                       │
│  ┌─────────────────────┐      ┌─────────────────────────────────────┐   │
│  │  IndexingService    │      │      RetrievalService               │   │
│  │  - chunk_document() │      │  - retrieve() → ports.Retriever     │   │
│  │  - index_book()     │      │  - rerank()   → ports.Reranker      │   │
│  │                     │      │  - generate() → ports.LLM           │   │
│  └─────────────────────┘      └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
          │                                    │
          ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              Ports (Interfaces)                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────────┐  │
│  │EbookParser  │ │VectorStore  │ │Embedder     │ │Retriever/Reranker │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
          │                │                │              │
          ▼                ▼                ▼              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            Adapters (Implementations)                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────────┐  │
│  │PdfParser    │ │ChromaAdapter│ │HF Embedder  │ │ChromaRetriever    │  │
│  │EpubParser   │ │             │ │             │ │CrossEncoderRanker │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- [Podman](https://podman.io/) (for integration tests)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ebook-rag-explorer
```

2. Create virtual environment and install dependencies:
```bash
uv sync
```

3. Copy environment template and configure:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Configuration

Edit `.env` file with your settings:

```bash
# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Reranking Model
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=your-api-key-here

# For LM Studio (local)
# LLM_PROVIDER=lmstudio
# LLM_BASE_URL=http://localhost:1234/v1

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_data

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval
RETRIEVAL_TOP_K=20
RERANK_TOP_N=5

# API
API_HOST=0.0.0.0
API_PORT=8000
```

## Running the Application

Start the FastAPI server:

```bash
# Development (with auto-reload)
uv run uvicorn ebook_rag_explorer.api.app:create_app --reload --host 0.0.0.0 --port 8000

# Or using tox
tox -e serve

# Production
tox -e serve-prod
```

The API will be available at `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## API Endpoints

- `POST /api/index` - Upload and index a PDF or EPUB file
- `POST /api/search` - Search indexed documents with RAG
- `GET /api/books` - List all indexed books
- `DELETE /api/books/{book_id}` - Delete a book from the index

## Testing

### Unit Tests
```bash
tox -e unit
```

### Integration Tests (requires Podman)
```bash
tox -e integration
```

### All Tests
```bash
tox -e all
```

### Podman Configuration

The project uses testcontainers with Podman. Configure your Podman socket:

**Linux:**
```bash
# Already configured in tox.ini, verify socket exists:
ls /run/user/1000/podman/podman.sock
```

**macOS:**
```bash
# Set DOCKER_HOST environment variable
export DOCKER_HOST=unix://$HOME/.local/share/containers/podman/machine/podman-machine-default/podman.sock
```

**Windows:**
```bash
set DOCKER_HOST=npipe:////./pipe/podman-machine-default
```

## Development

### Code Quality
```bash
# Linting
tox -e lint

# Type checking
tox -e type

# Fix linting issues
tox -e lint-fix
```

### Project Structure
```
src/ebook_rag_explorer/
├── config.py              # Application settings
├── models.py              # Pydantic models
├── ports/                 # Abstract interfaces
├── adapters/              # Concrete implementations
│   ├── parsers/           # PDF/EPUB parsers
│   ├── vectorstore/       # ChromaDB adapter
│   ├── embedding/         # Sentence-transformers
│   ├── retrieval/         # Retriever & reranker
│   └── llm/               # LLM adapters
├── services/              # Business logic
└── api/                   # FastAPI application
```

## License

MIT License
