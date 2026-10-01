# Research Notes: Benchmark Datasets for Academic Paper Processing

This document reviews the benchmark academic datasets evaluated and utilized for training and validating components of the SmartPaperAI pipeline.

---

## 1. Scientific Document Summarization Datasets

### 1.1 arXiv & PubMed Summarization (Cohan et al., 2018)
- **Source**: Scientific articles from arXiv.org and PubMed Central.
- **Size**: ~215,000 arXiv papers, ~133,000 PubMed papers.
- **Structure**: Includes full-text sections paired with author-written abstracts as ground-truth target summaries.
- **Relevance**: Serves as the primary reference domain for evaluating long-form academic document summarization. DistilBART demonstrated strong cross-domain transferability when evaluated on the introduction and conclusion sections of these corpuses.

### 1.2 SciTLDR (Cachola et al., 2020)
- **Source**: OpenReview papers with author-written and peer-reviewer-written single-sentence summaries.
- **Size**: ~3,229 papers, ~5,400 TLDR summaries.
- **Relevance**: Informed our "Key Insight & Simple Explanation" design, where complex multi-page contributions are distilled down to an intuitive, layperson summary.

---

## 2. Question Answering & Reading Comprehension Datasets

### 2.1 SQuAD 2.0 (Rajpurkar et al., 2018)
- **Content**: 150,000+ questions on Wikipedia articles, combining answerable questions with 50,000 unanswerable questions formulated by crowdworkers.
- **Relevance**: The core weights of `deepset/roberta-base-squad2` are pre-trained on SQuAD 2.0. The ability to handle "no-answer" or low-confidence thresholds prevents the model from hallucinating false answers when a user asks a question not addressed in the uploaded paper.

### 2.2 QASper (Dasigi et al., 2021)
- **Source**: NLP papers from the ACL Anthology.
- **Content**: Questions posed by NLP practitioners reading full papers, with answers grounded in text evidence spans, tables, or figures.
- **Relevance**: Validates our section-grounded RAG approach where exact source paragraphs and page numbers accompany the answer.
