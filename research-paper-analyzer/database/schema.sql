-- =============================================================================
-- SmartPaperAI — Database Schema
-- Compatible with SQLite and PostgreSQL
-- =============================================================================

-- 1. Documents Table: Stores uploaded research paper metadata and full text
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(64) PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(512) NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    page_count INTEGER NOT NULL,
    title VARCHAR(512),
    authors TEXT,                 -- Comma-separated or JSON array of author names
    doi VARCHAR(128),
    publication_year INTEGER,
    abstract TEXT,
    raw_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Document Sections Table: Detected academic sections with page references
CREATE TABLE IF NOT EXISTS document_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id VARCHAR(64) NOT NULL,
    section_name VARCHAR(64) NOT NULL, -- abstract, introduction, methodology, results, discussion, conclusion, references
    heading_text VARCHAR(255),
    page_number INTEGER,
    start_char_idx INTEGER,
    end_char_idx INTEGER,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sections_document_id ON document_sections(document_id);
CREATE INDEX IF NOT EXISTS idx_sections_name ON document_sections(section_name);

-- 3. Document Chunks Table: Segmented text chunks with embedding metadata for RAG QA
CREATE TABLE IF NOT EXISTS document_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id VARCHAR(64) NOT NULL,
    chunk_index INTEGER NOT NULL,
    section_name VARCHAR(64),
    page_number INTEGER,
    chunk_text TEXT NOT NULL,
    token_count INTEGER,
    embedding_model VARCHAR(128) DEFAULT 'all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_index ON document_chunks(chunk_index);

-- 4. Document Analyses Table: AI-generated summaries, insights, and research directions
CREATE TABLE IF NOT EXISTS document_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id VARCHAR(64) NOT NULL UNIQUE,
    executive_summary TEXT NOT NULL,
    research_objective TEXT,
    methodology TEXT,
    key_findings TEXT,
    key_insight TEXT,
    simple_explanation TEXT,
    limitations TEXT,
    keywords TEXT,           -- JSON array of keywords and relevance scores
    main_points TEXT,        -- JSON array of top ranked extractive sentences
    future_research TEXT,    -- JSON array of proposed future directions with evidence
    model_versions TEXT,     -- JSON object of models used (summarizer, QA, embedder)
    processing_time_sec REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_analyses_document_id ON document_analyses(document_id);

-- 5. QA Conversations Table: Chat history with citations and confidence scores
CREATE TABLE IF NOT EXISTS qa_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id VARCHAR(64) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    confidence_score REAL,
    source_section VARCHAR(64),
    source_page INTEGER,
    context_snippet TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_qa_document_id ON qa_messages(document_id);
