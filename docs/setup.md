# Getting started

## Requirements

- Python 3.12
- A **Nebius AI Studio API key** (`NEBIUS_API_KEY`). Ingestion, every question and evaluation are **paid** calls (small, but not free)
- Docker, only for Opik (optional)
- Disk: ~1.2 GB for the dataset download

Ports: **8000** API, **5500** chat page, **5173** Opik.

## Steps

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                 # set NEBIUS_API_KEY
python -m scripts.download_data
python -m scripts.sample_catalog --size 2000
python -m src.ingestion              # shows a cost estimate and asks to confirm
python main.py                   # API
python -m http.server 5500           # chat page: http://localhost:5500/public/
scripts/opik.sh up                   # optional: Opik on :5173
python -m src.evaluation.evaluator --retrieval-only   # evaluation (paid embeddings)
npx docsify-cli serve docs           # these docs
```

## Troubleshooting

- **503 "NEBIUS_API_KEY is not set"**: fill `.env`.
- **503 "vector store is empty"**: run `python -m src.ingestion`.
- **`ModuleNotFoundError: src`**: run modules from the repo root with `python -m`.
- **Chat page says "Failed to fetch"**: the API is not running on port 8000.
- **Ingestion stopped halfway or `429` errors**: embeddings are sent in batches of 100 with retries and backoff to avoid the rate limit (see [Data](data.md)). Run it again, finished products are skipped.
