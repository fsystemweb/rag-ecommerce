"""FastAPI app. Run with: python main.py"""

import json
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.core import config
from src.core.models import AskRequest, AskResponse


@asynccontextmanager
async def lifespan(_):
    from src.observability.opik_sync import seed_prompts

    seed_prompts()  # no-op without Opik
    yield


app = FastAPI(title="rag-ecommerce", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_pipeline():
    """Built on the first request, so /health works without a key or a store."""
    from src.agent.rag_agent import Pipeline
    from src.agent.retriever import Retriever
    from src.ingestion.index import get_store

    if not config.NEBIUS_API_KEY:
        raise HTTPException(503, "NEBIUS_API_KEY is not set. Copy .env.example to .env and fill it.")
    store = get_store()
    if store._collection.count() == 0:
        raise HTTPException(503, "The vector store is empty. Run: python -m src.ingestion")
    return Pipeline(Retriever(store))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline=Depends(get_pipeline)) -> AskResponse:
    return pipeline.ask(req)


@app.post("/ask/stream")
def ask_stream(req: AskRequest, pipeline=Depends(get_pipeline)):
    """Server-Sent Events: status, token, reset and final events."""

    def events():
        for e in pipeline.run(req):
            if e["type"] == "final":
                e = {"type": "final", "response": e["response"].model_dump()}
            yield f"data: {json.dumps(e, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000)
