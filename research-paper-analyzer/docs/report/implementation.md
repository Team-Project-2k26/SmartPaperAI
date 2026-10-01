# Academic Project Report — Section 3: Implementation Architecture

## 3.1 System Components
The project is built as a modular micro-monolith cleanly decoupled into three primary tiers:

```
┌────────────────────────────────────────────────────────┐
│                   React + Vite SPA                     │
│  (Modern Dark UI, Drag-and-Drop Upload, Streaming UX)  │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP / JSON
┌───────────────────────────▼────────────────────────────┐
│                    FastAPI Backend                     │
│  • Pydantic v2 validation • Settings Management       │
│  • Asynchronous handlers  • CORS Middleware            │
└───────┬───────────────────┬────────────────────┬───────┘
        │                   │                    │
┌───────▼────────┐  ┌───────▼────────┐   ┌───────▼───────┐
│ Document Proc  │  │ Summarization  │   │   QA Model    │
│  • PyMuPDF     │  │  • TF-IDF Rank │   │ • MiniLM RAG  │
│  • Text Clean  │  │  • DistilBART  │   │ • RoBERTa QA  │
│  • Chunker     │  │  • YAKE Keywrd │   │ • Span Extr.  │
└────────────────┘  └────────────────┘   └───────────────┘
```

## 3.2 Service Layer Modules
- `backend/app/services/document_service.py`: Orchestrates PDF ingestion, file validation, storage, and section extraction.
- `backend/app/services/summarization_service.py`: Interfaces with the NLP preprocessing pipeline, extractive ranking, and HuggingFace DistilBART inference.
- `backend/app/services/qa_service.py`: Manages vector retrieval via sentence-transformers and extractive QA span answering via RoBERTa.

## 3.3 Concurrency & Memory Management
- **Lazy Loading with Startup Warmup**: Models are loaded lazily as singletons upon first access, with optional background pre-warming at application startup via FastAPI lifespan hooks.
- **CPU Offloading & Batching**: Tokenization and inference operate with dynamic sequence truncation (`max_length=512` or `max_length=1024`) to stay within modest 4GB RAM footprints on consumer laptops.
- **In-Memory Cache**: Active documents, parsed chunks, and calculated embedding matrices are stored in a session cache to eliminate duplicate model calls during interactive Q&A.
