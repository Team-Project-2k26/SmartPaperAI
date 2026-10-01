# SmartPaperAI — API Documentation

This document describes the RESTful API endpoints exposed by the SmartPaperAI backend service.

**Base URL**: `http://localhost:8000`  
**Interactive Docs**: `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc` (ReDoc)

---

## 1. System Endpoints

### `GET /health`
Verifies backend service health, version, and the loading state of NLP models.

- **Request**: None
- **Response** `200 OK`:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "models": {
    "qa": "loaded"
  }
}
```

---

## 2. Document Ingestion

### `POST /upload`
Uploads a research paper in PDF format. Extracts raw text via PyMuPDF, cleans noise, parses structural academic sections, extracts metadata (title, authors, DOI, year), splits content into overlapping sentence chunks, and indexes embeddings.

- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `file`: PDF file binary (`.pdf` extension required, max size 50 MB)
- **Response** `200 OK`:
```json
{
  "document_id": "9f8b2c4e-1234-5678-abcd-ef0123456789",
  "filename": "vaswani2017.pdf",
  "page_count": 15,
  "detected_sections": [
    "abstract",
    "introduction",
    "background",
    "methodology",
    "experiments",
    "results",
    "conclusion",
    "references"
  ],
  "message": "File uploaded and parsed successfully."
}
```
- **Error Responses**:
  - `400 Bad Request`: Non-PDF file uploaded, file empty, or file exceeds size limit.
  - `422 Unprocessable Entity`: Corrupt PDF file or text extraction yielded fewer than 50 characters.

---

## 3. Analysis Pipeline

### `POST /analyze`
Runs the complete multi-stage AI analysis pipeline on an uploaded document:
1. Abstractive summary via DistilBART (`sshleifer/distilbart-cnn-12-6`)
2. Key research objective extraction
3. Methodology synthesis
4. Experimental findings extraction
5. Key insight extraction and simple layperson explanation
6. Keyword extraction via YAKE
7. Known limitations identification
8. Grounded future research directions generation (minimum 3)
9. Extractive sentence ranking (TF-IDF + Cosine similarity + Position weighting)

- **Request Body** `application/json`:
```json
{
  "document_id": "9f8b2c4e-1234-5678-abcd-ef0123456789"
}
```
- **Response** `200 OK`:
```json
{
  "document_id": "9f8b2c4e-1234-5678-abcd-ef0123456789",
  "summary": "This paper presents the Transformer architecture based solely on self-attention...",
  "objective": "To develop a sequence transduction model that dispenses with recurrence and convolutions...",
  "methodology": "The Transformer uses stacked self-attention and point-wise, fully connected layers...",
  "key_findings": "The model achieved 28.4 BLEU on English-to-German, outperforming existing best models...",
  "key_insight": "Self-attention mechanisms alone are sufficient for state-of-the-art sequence modeling.",
  "simple_explanation": "Rather than reading text one word at a time, the model looks at every word at the exact same time and measures how each word relates to all others.",
  "keywords": [
    {"keyword": "transformer", "score": 0.012},
    {"keyword": "self-attention", "score": 0.018},
    {"keyword": "multi-head attention", "score": 0.024}
  ],
  "limitations": "Computational memory scales quadratically with input sequence length.",
  "future_research": [
    "Investigate linear and sparse attention variants to handle longer document contexts.",
    "Apply attention mechanisms to non-textual modalities including images and audio.",
    "Explore restricted attention spans for local dependency tasks."
  ],
  "main_points": [
    "The Transformer relies entirely on attention mechanisms.",
    "Training took 3.5 days on 8 P100 GPUs.",
    "Establishes a new state of the art in machine translation."
  ]
}
```

---

## 4. Question & Answering (RAG)

### `POST /chat`
Accepts a natural-language query regarding an analyzed document. Computes query embeddings, retrieves top-k semantically relevant chunks from the paper text, and runs extractive QA inference via RoBERTa (`deepset/roberta-base-squad2`).

- **Request Body** `application/json`:
```json
{
  "document_id": "9f8b2c4e-1234-5678-abcd-ef0123456789",
  "question": "What is the purpose of positional encodings?"
}
```
- **Response** `200 OK`:
```json
{
  "answer": "Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens.",
  "confidence_score": 0.89,
  "sources": [
    {
      "section": "methodology",
      "page": 6,
      "snippet": "Positional Encoding: Since our model contains no recurrence and no convolution..."
    }
  ]
}
```
