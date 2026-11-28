# MereRunTextUtilities

Text-utility and embedding runtimes carved out of `MereRunCore`:

- `Embeddings/`: EmbeddingGemma 2 (MLX + Core ML via `MereRunCoreMLModel`), PPLX-Embed v2, and Qwen3-Embedding.
- `PrivacyFilter/`: PII redaction and anonymization.
- `Clef/`: structured classification and omni routing (`AudioQwen3ASRModel` is scoped here).
- `GLiNER/`, `D1/`, and `Laya/`: entity extraction, decision classification, and text utility runtimes (`MereRunGLiNERModel`, `MereRunD1Model`, and `MereRunLayaModel` are scoped to this target).

Catalog metadata and resource validation stay in `MereRunCore` so Core and iOS Studio no longer depend on these 5 specialist model targets.
