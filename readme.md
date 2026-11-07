# rag-ecommerce

An educational **product search assistant** built with RAG: it answers only from a catalog of thousands of Spanish-language products (Amazon ESCI), cites the products it recommends, checks its own answer with an LLM judge, and is measured with a labeled golden dataset.

⚠️ Educational project, not production-ready. Using it calls the OpenAI API and costs a little money (ingestion, each question, evaluation).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # set OPENAI_API_KEY
python -m scripts.download_data # ~1 GB, once
python -m scripts.sample_catalog --size 10000
python -m app.ingestion         # embeds the catalog (paid, asks to confirm)
python -m app.main              # API on :8000
python -m http.server 5500      # chat page: http://localhost:5500/public/
```

Try it: `curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question":"auriculares inalámbricos para correr"}'`

## Learn more

Full docs (overview, data, architecture, evaluation, FAQ): see [`docs/`](docs/) — `npx docsify-cli serve docs`.
Contributor rules: [`AGENTS.md`](AGENTS.md). Tests: `pytest`.
