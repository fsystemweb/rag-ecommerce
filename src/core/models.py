"""Pydantic models used by the API."""

from pydantic import BaseModel, Field


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
