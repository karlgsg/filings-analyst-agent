CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    doc_name    TEXT NOT NULL,
    company     TEXT,
    doc_type    TEXT,           -- 10k, 10q, 8k, earnings
    doc_year    INT,
    page        INT NOT NULL,   -- 1-indexed
    chunk_index INT NOT NULL,   -- position within the page
    text        TEXT NOT NULL,
    embedding   vector(384) NOT NULL,   -- bge-small-en-v1.5
    -- Full-text search column, kept in sync automatically by Postgres.
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (doc_name, page, chunk_index)
);

-- Approximate nearest-neighbour index for cosine distance (the <=> operator).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Inverted index for full-text search (used in Task 8).
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS chunks_company_year ON chunks (company, doc_year);
