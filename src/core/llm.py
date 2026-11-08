"""Model clients. Nebius AI Studio speaks the OpenAI API, so langchain-openai is used with its base URL."""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.core import config


def make_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=config.NEBIUS_API,
        api_key=config.NEBIUS_API_KEY,
        temperature=0,
        timeout=30,
        max_retries=2,
        stream_usage=True,
    )


def make_embeddings() -> OpenAIEmbeddings:
    # check_embedding_ctx_length=False sends plain text (not OpenAI token ids), which other providers require
    return OpenAIEmbeddings(
        model=config.EMBEDDINGS_MODEL, base_url=config.NEBIUS_API, api_key=config.NEBIUS_API_KEY, check_embedding_ctx_length=False
    )
