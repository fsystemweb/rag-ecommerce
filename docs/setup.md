# Getting started

## Requirements

- Python 3.12
- An **OpenAI API key**. Ingestion, every question and evaluation are **paid** calls (small, but not free)
- Docker, only for Opik (optional)
- Disk: ~1.2 GB for the dataset download

Ports: **8000** API, **5500** chat page, **5173** Opik.

## Steps

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                 # set OPENAI_API_KEY
python -m scripts.download_data
python -m scripts.sample_catalog --size 10000
python -m app.ingestion              # shows a cost estimate and asks to confirm
python -m app.main                   # API
python -m http.server 5500           # chat page: http://localhost:5500/public/
scripts/opik.sh up                   # optional: Opik on :5173
python -m evaluation.run --retrieval-only   # evaluation (paid embeddings)
npx docsify-cli serve docs           # these docs
```

To try without OpenAI: `python -m evaluation.run --search bm25 --retrieval-only` needs only the catalog.

## Troubleshooting

- **503 "OPENAI_API_KEY is not set"**: fill `.env`.
- **503 "vector store is empty"**: run `python -m app.ingestion`.
- **`ModuleNotFoundError: app`**: run modules from the repo root with `python -m`.
- **Chat page says "Failed to fetch"**: the API is not running on port 8000.
- **Ingestion stopped halfway**: run it again, finished products are skipped.
