# Contributing, cost and safety

Rules live in [`AGENTS.md`](https://github.com/fsystemweb/rag-ecommerce/blob/main/AGENTS.md): keep it simple, imports with `src.`, settings only in `src/core/config.py`, prompts only in `src/agent/prompts/`, tests offline (`pytest`), lint with `ruff`. Ask before adding a dependency, changing the `/ask` shape or the golden dataset.

## Cost and safety

- Each question calls Nebius **2 to 3 times** (classifier, rewriter for follow-ups only, generator) plus one embedding, with 10 products in the prompt.
- Ingestion (embeddings) and evaluation also cost money. Never run them in CI on pull requests.
- Product text is treated as data, questions are checked for prompt injection, and the system prompt is protected by a canary check.
- Educational project: not production-ready.

## Publish on GitHub Pages

Settings → Pages → Deploy from branch → `main` / `docs`. Expected URL: https://fsystemweb.github.io/rag-ecommerce/
