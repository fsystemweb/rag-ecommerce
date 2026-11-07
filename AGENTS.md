# AGENTS.md

`rag-ecommerce` is an educational product-search assistant: FastAPI + LangChain + OpenAI over a sampled Spanish Amazon ESCI catalog (RAG with hybrid search, a grounding judge, Opik tracing and an offline evaluation). Not production-ready. Keep it simple: a student must be able to run it locally.

## Commands
- Setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt`, then `cp .env.example .env` and set `OPENAI_API_KEY`
- Data (one time): `python -m scripts.download_data` then `python -m scripts.sample_catalog --size 10000`
- Ingest (paid embeddings): `python -m app.ingestion [--limit N] [--yes]`
- API: `python -m app.main` (port 8000)
- Chat page: `python -m http.server 5500`, open `http://localhost:5500/public/`
- Tests (offline, free): `pytest` and `python -m evaluation.test_metrics`
- Lint: `ruff check . && ruff format --check .` (CI runs the same plus `pytest`)
- Docs: `npx docsify-cli serve docs`

## Conventions
- `app/` code, `app/prompts/*.txt` prompts (never inline), `scripts/` data tools, `evaluation/` golden set and metrics, `public/` static chat page, `tests/` offline tests.
- Always import with the `app.` prefix and run modules with `python -m`. No code depends on the working directory.
- All settings are read in `app/config.py` only.

## Boundaries
- Always: keep tests offline (fake LLM and fake embeddings).
- Ask first: adding a dependency, changing the `/ask` request or response shape, changing the judge retry threshold, changing the golden dataset.
- Never: commit secrets or the dataset, read `.env`, or run `ingestion`/`evaluation.run` without being asked (they cost money).
