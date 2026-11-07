# Observability

With [Comet Opik](https://github.com/comet-ml/opik) running (`scripts/opik.sh up`, UI on port 5173) and `OPIK_URL_OVERRIDE` set, each question is a **trace**: topic classification, rewrite, retrieval (with dense score and BM25 rank), and generation, with the prompt versions in the metadata. Prompts are read from Opik first (`rag-ecommerce-generator`, `-rewriter`, `-classifier`), with `src/agent/prompts/*.txt` as fallback and seed. Upload the golden set with `python -m src.evaluation.dataset --upload`; evaluation runs appear as **experiments**.

> Screenshots (traces, retriever span, prompts, dataset, experiments) are **pending**: they need a running Opik and a Nebius key, which were not available when these docs were written. Add them to `docs/assets/` with a one-line caption each.
