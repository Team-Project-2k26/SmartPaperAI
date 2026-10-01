# Academic Project Report — Section 2: Methodology

## 2.1 PDF Parsing & Layout Normalization
The ingestion pipeline leverages PyMuPDF (`fitz`) for fast, memory-safe text and layout extraction from academic PDFs. The extraction pipeline performs:
1. **Font & Size Analysis**: Identifies headings vs. body text based on font size heuristics and style tags.
2. **De-hyphenation & Ligature Cleaning**: Merges words broken across lines by end-of-line hyphens (e.g., `transfor-\nmation` -> `transformation`) and expands Unicode ligatures (fi, fl, ffi).
3. **Artifact Removal**: Strips running headers, footers, page numbering, and line numbers using positional bounding box filtering.

## 2.2 Section Detection Heuristics
Academic manuscripts generally follow the IMRaD (Introduction, Methods, Results, and Discussion) convention. A multi-pattern regex engine combined with layout signals categorizes text blocks into canonical sections:
- `abstract`: Primary problem statement and high-level findings.
- `introduction`: Motivation, background, and research contributions.
- `methodology` / `approach`: Algorithmic models, architectures, and theoretical derivations.
- `experiments` / `results`: Datasets, metrics (e.g., BLEU, F1, Accuracy), and empirical benchmarks.
- `discussion` / `limitations`: Critical analysis of weaknesses, assumptions, and threats to validity.
- `conclusion`: High-level synthesis and future research directions.

## 2.3 Sentence Scoring & Hybrid Extractive Ranking
To prevent loss of critical numerical metrics and factual contributions, SmartPaperAI combines statistical TF-IDF ranking with semantic embedding centroid similarity:
$$S(s_i) = \alpha \cdot \text{TF-IDF}(s_i) + \beta \cdot \cos(\mathbf{e}_i, \mathbf{c}) + \gamma \cdot \text{Pos}(s_i)$$
where:
- $\text{TF-IDF}(s_i)$ rewards domain-specific keywords with high document salience.
- $\cos(\mathbf{e}_i, \mathbf{c})$ computes the cosine similarity between sentence embedding $\mathbf{e}_i$ and the document centroid embedding $\mathbf{c}$ generated via `all-MiniLM-L6-v2`.
- $\text{Pos}(s_i)$ applies lead bias, prioritizing lead sentences in sections.

## 2.4 Abstractive Summarization with DistilBART
For abstractive summarization, the system utilizes `sshleifer/distilbart-cnn-12-6`, a student distillation of BART-large trained on CNN/DailyMail. It produces concise summaries within a 142-token ceiling while maintaining factual fidelity.

## 2.5 Retrieval-Augmented Question Answering
For interactive querying:
1. Text is chunked with a sliding window of 250 tokens and 50 tokens overlap.
2. Each chunk is indexed with dense vector embeddings (`all-MiniLM-L6-v2`).
3. User queries trigger cosine similarity retrieval against chunk vectors to return top-$k$ candidates.
4. The retrieved context is passed to `deepset/roberta-base-squad2`, which outputs start and end span token logits representing the exact factual answer within the source text, accompanied by confidence probability.
