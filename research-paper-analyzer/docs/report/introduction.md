# Academic Project Report — Section 1: Introduction

## 1.1 Context and Motivation
Scientific and scholarly literature has been growing at an exponential rate. With hundreds of thousands of research papers published annually across arXiv, IEEE, ACM, PubMed, and bioRxiv, researchers, graduate students, and industry engineers face severe information overload. Identifying relevant papers, extracting the key methodological contributions, evaluating experimental outcomes, and synthesizing future research directions from unstructured 10-to-30 page PDF documents requires hours of tedious manual reading.

## 1.2 Problem Formulation
Existing commercial solutions suffer from three significant drawbacks:
1. **Proprietary & Expensive APIs**: Rely heavily on commercial closed APIs (such as OpenAI GPT-4 or Anthropic Claude) which introduce recurring financial costs, quota restrictions, and data privacy concerns.
2. **Hallucination Risk**: Large generative models often hallucinate findings or cite phantom sections when not strictly grounded in the document structure.
3. **Black-box Summaries**: Unstructured paragraph outputs fail to highlight critical academic dimensions such as the core objective, experimental methodology, key insight, and proposed future directions.

## 1.3 Project Goals & Objectives
SmartPaperAI solves these challenges by providing a 100% open-source, locally runnable academic research paper understanding platform. The primary engineering goals are:
- **Zero API Dependency**: Run all neural models (embeddings, summarization, and question answering) locally on CPU or GPU without external cloud calls.
- **Academic Structural Decomposition**: Automatically partition PDFs into standard academic sections (Abstract, Introduction, Methodology, Experiments, Results, Discussion, Conclusion).
- **Structured Multi-Task Output**: Provide a clean breakdown covering executive summary, research objective, method summary, key findings, layman-friendly key insight, keywords, limitations, and actionable future research questions.
- **Grounded Q&A via RAG**: Implement Retrieval-Augmented Generation using dense semantic retrieval with exact source section and page citations.
