"""Pydantic models shared by the API, the ingestion and the evaluation."""

from pydantic import BaseModel, Field


class Product(BaseModel):
    product_id: str
    title: str
    description: str = ""
    bullets: str = ""
    brand: str = ""
    color: str = ""


class Turn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[Turn] = Field(default_factory=list)


class ProductRef(BaseModel):
    product_id: str
    title: str
    brand: str = ""


class AskResponse(BaseModel):
    answer: str
    products: list[ProductRef] = Field(default_factory=list)
    grounded: bool = True
    attempts: int = 1
