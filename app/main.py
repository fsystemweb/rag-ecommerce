"""FastAPI app. Run with: python -m app.main"""

from fastapi import FastAPI

from app.models import AskRequest, AskResponse

app = FastAPI(title="rag-ecommerce")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    # Skeleton: the real chain arrives with the generation ticket.
    return AskResponse(answer="Not implemented yet.", grounded=False, attempts=0)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
