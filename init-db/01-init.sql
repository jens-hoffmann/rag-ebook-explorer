-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- Books table (must exist before documents due to FK)
CREATE TABLE IF NOT EXISTS books (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    author VARCHAR(255),
    format VARCHAR(50),
    collection_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    chunk_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for collection lookups
CREATE INDEX IF NOT EXISTS idx_books_collection ON books(collection_id);

-- Documents table with hybrid search support
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) NOT NULL UNIQUE,
    book_id VARCHAR(255) NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    collection_id VARCHAR(255),
    
    -- Content
    content TEXT NOT NULL,
    
    -- Vector embedding (dimension 384 for all-MiniLM-L6-v2)
    -- Note: dimension can be adjusted based on the embedding model used
    embedding VECTOR(384),
    
    -- Full-text search vector (generated automatically)
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', COALESCE(content, '')), 'A')
    ) STORED,
    
    -- Chunk metadata
    chunk_index INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    
    -- Source metadata stored as JSONB
    source_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_book_id ON documents(book_id);
CREATE INDEX IF NOT EXISTS idx_documents_collection_id ON documents(collection_id);
CREATE INDEX IF NOT EXISTS idx_documents_chunk_id ON documents(chunk_id);

-- Vector index using ivfflat for cosine similarity
CREATE INDEX IF NOT EXISTS idx_documents_embedding 
ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Full-text search GIN index
CREATE INDEX IF NOT EXISTS idx_documents_search 
ON documents USING GIN (search_vector);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to auto-update updated_at
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to update book chunk count
CREATE OR REPLACE FUNCTION update_book_chunk_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE books SET chunk_count = chunk_count + 1 WHERE id = NEW.book_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE books SET chunk_count = chunk_count - 1 WHERE id = OLD.book_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

-- Trigger to maintain book chunk count
DROP TRIGGER IF EXISTS update_book_chunk_count_trigger ON documents;
CREATE TRIGGER update_book_chunk_count_trigger
    AFTER INSERT OR DELETE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_book_chunk_count();

-- View for collection statistics
CREATE OR REPLACE VIEW collection_stats AS
SELECT 
    collection_id AS id,
    collection_id AS name,
    COUNT(DISTINCT book_id) AS book_count,
    COUNT(*) AS chunk_count
FROM documents
WHERE collection_id IS NOT NULL
GROUP BY collection_id
ORDER BY collection_id;

-- View for book details with collection info
CREATE OR REPLACE VIEW book_details AS
SELECT 
    b.id,
    b.title,
    b.author,
    b.format,
    b.collection_id,
    b.metadata,
    b.chunk_count,
    b.created_at,
    COUNT(DISTINCT d.collection_id) as collection_count
FROM books b
LEFT JOIN documents d ON b.id = d.book_id
GROUP BY b.id, b.title, b.author, b.format, b.collection_id, b.metadata, b.chunk_count, b.created_at;

-- Comment on embedding dimension
COMMENT ON TABLE documents IS 'Document chunks with vector embeddings (dimension 384 for all-MiniLM-L6-v2). Adjust VECTOR dimension if using a different model.';
