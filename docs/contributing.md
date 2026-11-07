# Contributing, cost and safety

Rules live in [`AGENTS.md`](https://github.com/fsystemweb/rag-ecommerce/blob/main/AGENTS.md): keep it simple, imports with `app.`, settings only in `app/config.py`, prompts only in `app/prompts/`, tests offline (`pytest`), lint with `ruff`. Ask before adding a dependency, changing the `/ask` shape, the judge threshold or the golden dataset.

## Cost and safety

- Each question calls Nebius up to **3 times** (generator + judge per attempt) with 5 products in the prompt.
- Ingestion (embeddings) and evaluation also cost money. Never run them in CI on pull requests.
- Product text is treated as data, questions are checked for prompt injection, and the system prompt is protected by a canary check.
- Educational project: not production-ready.

## Publish on GitHub Pages

Settings → Pages → Deploy from branch → `main` / `docs`. Expected URL: https://fsystemweb.github.io/rag-ecommerce/
