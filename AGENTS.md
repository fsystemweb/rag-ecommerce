# AGENTS.md

`rag-ecommerce` is an educational Retrieval-Augmented Generation chatbot for e-commerce: it embeds a Markdown product file and answers customer questions with an OpenAI-compatible LLM API. Not production-ready.

## Commands
- Create venv: `python3 -m venv env`, then activate it with `source env/bin/activate` (Python 3.12.3 or newer per the readme)
- Install: `pip install -r requirements.txt`
- Web UI: from the repo root, run `streamlit run src/ui/app.py` with `PYTHONPATH` set to the repo root
- Build embeddings: `python model_pipeline.py` from inside `src/` (with `PYTHONPATH` set to the repo root); it reads `../data/data.md` and writes `model_data.json` into the current directory.
- Terminal chat: `python query_executor.py` from inside `src/` (same `PYTHONPATH` note); it starts an interactive loop.
- Validation and metrics: `src/validation.py`; analysis scripts live in `doc/`.
- Tests and lint: none yet

## Conventions
- Python with pip and `requirements.txt` (PyMuPDF, numpy, openai, matplotlib, networkx, seaborn). The code also imports `streamlit`, `requests` and `dotenv`, which are not listed there; `requirements.txt` has `load_dotenv`, which is not the package name (`python-dotenv`).
- `src/config/openai_client.py`: OpenAI client pointed at a custom base URL. `src/model_pipeline.py`: chunking and embeddings. `src/query_executor.py`: retrieval by cosine similarity and answer generation. `src/util/generate_response.py`: LLM call and similarity helpers. `src/ui/app.py`: Streamlit UI.
- `src/validation.py` imports without the `src.` prefix (`from query_executor import ...`), unlike the other modules; run it from `src/`.
- `data/data.md` is the product knowledge base, `data/template.md` the template for it, `data/val.json` validation questions. `data/raw-data/` is git-ignored.
- `model_data.json` is generated embeddings output (about 750 KB) committed at the repo root.
- `metrics/` holds per-model evaluation logs and charts. Code comments and some data are in Spanish.
- Configuration comes from a git-ignored `.env` file loaded with `load_dotenv()`. No `.env.example` exists in the repo.

## Boundaries
- Always: keep new modules under `src/` and import them with the `src.` prefix.
- Ask first: adding a dependency, changing the embedding model, regenerating `model_data.json`.
- Never: commit secrets or read .env files.
