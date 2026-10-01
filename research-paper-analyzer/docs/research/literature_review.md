# Research Notes: Literature Review

A review of fundamental literature shaping scientific document understanding, abstractive summarization, and retrieval-augmented question answering.

---

## 1. Document Extraction & Structure Recovery
- **Ramakrishnan et al. (2012)**: *Layout-aware text extraction from scientific publications.* Investigated coordinate-based geometric segmentation of multi-column layouts. Established the foundation for treating font-size differentials as strong signals for section heading detection.
- **Tkaczyk et al. (2015)**: *CERMINE — automatic extraction of structured metadata from scientific articles.* Explored rule-based and SVM-based layout parsers. Highlighted the efficiency advantages of combining fast regex parsers with geometric bounding boxes over heavy neural vision approaches for standard text-rich PDFs.

---

## 2. Text Summarization Architectures
- **Lewis et al. (2020)**: *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.* Introduced bidirectional encoder with autoregressive decoder, proving ideal for summarization tasks by combining BERT's comprehension with GPT's generation.
- **Sanh et al. (2019) / Shleifer & Rush (2020)**: *Pre-trained Summarization Distillation.* Demonstrated that student models (such as `distilbart-cnn-12-6`) retain 95%+ of full BART-large performance while running 2x faster with half the parameter count (306M parameters vs 406M parameters), making local CPU inference viable.

---

## 3. Dense Retrieval and Span Question Answering
- **Reimers & Gurevych (2019)**: *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* Replaced pairwise cross-encoder evaluations with siamese networks, yielding dense vector representations capable of sub-millisecond cosine similarity searches across thousands of document chunks.
- **Liu et al. (2019)**: *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* Demonstrated that removing Next Sentence Prediction (NSP), training with dynamic masking, and training on larger mini-batches yields superior token representation for downstream span extraction tasks like SQuAD 2.0.
- **Lewis et al. (2020)**: *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* Showed that grounding generative or extractive models with non-parametric dense vector indices substantially reduces hallucinations and provides verifiable provenance.
