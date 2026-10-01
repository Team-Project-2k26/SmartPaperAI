# Research Notes: Model Selection & Architectural Trade-offs

This document details the evaluation criteria and rationale for the model selections in SmartPaperAI.

---

## 1. Design Constraints
1. **Local Execution**: Must run smoothly on standard consumer hardware (CPU or modest GPU, ≤8GB system RAM).
2. **Zero Paid APIs**: No reliance on third-party API services, billing accounts, or cloud tokens.
3. **Low Latency**: End-to-end paper ingestion in under 30 seconds; Q&A turnaround under 1.5 seconds.
4. **Factual Grounding**: Elimination of ungrounded hallucinations in answered queries.

---

## 2. Model Selection Rationale

| Pipeline Task | Candidate Models | Selected Model | Rationale & Trade-offs |
|---|---|---|---|
| **Semantic Embedding** | `all-MiniLM-L6-v2`<br>`all-mpnet-base-v2`<br>`bge-small-en-v1.5` | **`all-MiniLM-L6-v2`** | ~80 MB disk footprint, 384 embedding dimensions, 5x faster inference than mpnet-base with 92% comparable retrieval accuracy on semantic search tasks. |
| **Abstractive Summarization** | `facebook/bart-large-cnn`<br>`t5-base`<br>`sshleifer/distilbart-cnn-12-6` | **`sshleifer/distilbart-cnn-12-6`** | 1.2 GB download vs 1.6 GB for full BART; 2.1x higher throughput on CPU; preserves high ROUGE scores without excessive memory spikes. |
| **Question Answering** | `deepset/roberta-base-squad2`<br>`distilbert-base-cased-distilled-squad`<br>`google/flan-t5-base` | **`deepset/roberta-base-squad2`** | Extractive span architecture eliminates hallucination risk entirely. SQuAD 2.0 training enables calibrated "no-answer" detection when query is out of context. |
| **Keyword Extraction** | `KeyBERT`<br>`YAKE`<br>`RAKE` | **`YAKE`** | Unsupervised statistical approach requires zero neural model overhead, runs in <200ms, and provides robust multi-word n-gram keyword candidates. |
| **NLP Preprocessing** | `NLTK`<br>`spaCy en_core_web_sm` | **`spaCy` + `NLTK` fallback** | Accurate sentence boundary detection (SBD) across complex academic typography with graceful regex/NLTK fallbacks. |
