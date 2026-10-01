-- =============================================================================
-- SmartPaperAI — Seed Data for Local Development & Testing
-- =============================================================================

INSERT INTO documents (
    id, filename, file_path, file_size_bytes, page_count,
    title, authors, doi, publication_year, abstract, raw_text
) VALUES (
    'doc_seed_attention_1706',
    'attention_is_all_you_need.pdf',
    'uploads/attention_is_all_you_need.pdf',
    2215680,
    15,
    'Attention Is All You Need',
    'Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin',
    '10.48550/arXiv.1706.03762',
    2017,
    'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer, a model architecture eschewing recurrence and entirely relying on an attention mechanism to draw global dependencies between input and output.',
    'Full extracted text of the paper Attention Is All You Need...'
);

INSERT INTO document_analyses (
    document_id, executive_summary, research_objective, methodology,
    key_findings, key_insight, simple_explanation, limitations,
    keywords, main_points, future_research, processing_time_sec
) VALUES (
    'doc_seed_attention_1706',
    'This paper introduces the Transformer, an architecture based entirely on self-attention mechanisms that completely replaces recurrent and convolutional layers for sequence modeling tasks.',
    'To create a sequence transduction model that avoids sequential computation, enables massive parallelization, and captures long-range dependencies efficiently.',
    'Multi-Head Self-Attention coupled with sinusoidal positional encodings, layer normalization, and feed-forward networks organized in an encoder-decoder topology.',
    'The Transformer achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, establishing a new state of the art while training significantly faster than previous models.',
    'Self-attention mechanisms alone, without any recurrence or convolution, are sufficient for high-performance sequence transduction.',
    'Instead of reading a sentence word by word from start to finish like previous systems, the model looks at every word in relation to all other words at the exact same moment.',
    'Computational complexity scales quadratically O(n^2) with sequence length, making very long documents memory-intensive.',
    '["transformer", "self-attention", "sequence-to-sequence", "neural machine translation", "multi-head attention"]',
    '["The Transformer is the first transduction model relying entirely on self-attention.", "On the WMT 2014 English-to-German task, the Transformer achieves 28.4 BLEU.", "The model allows for significantly more parallelization and requires less time to train."]',
    '["Investigate linear-time or sparse attention variants to mitigate quadratic memory cost.", "Extend the Transformer architecture to non-text modalities such as audio, vision, and video.", "Explore conditional computation to dynamically route tokens and increase model capacity without proportional FLOP increases."]',
    12.4
);

INSERT INTO qa_messages (
    document_id, question, answer, confidence_score, source_section, source_page, context_snippet
) VALUES (
    'doc_seed_attention_1706',
    'What is Multi-Head Attention?',
    'Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions.',
    0.94,
    'methodology',
    4,
    'Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.'
);
