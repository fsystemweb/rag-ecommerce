# rag-ecommerce

A **product search and recommendation assistant**. You ask for what you need ("una funda negra para iPhone 13") and it answers **only from a catalog** of thousands of Spanish-language Amazon products, naming the products it recommends.

## Why it exists

It is an educational example of RAG (Retrieval-Augmented Generation) at a size where RAG makes sense (a few thousand products do not fit in one prompt), together with **evaluation** and **observability**.

## Try it

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "auriculares inalámbricos para correr"}'
```

```json
{
  "answer": "Te recomiendo Sony WF-... [B0ABC123]: son inalámbricos y ...",
  "products": [{"product_id": "B0ABC123", "title": "Sony WF-...", "brand": "Sony"}],
  "grounded": true,
  "attempts": 1
}
```

(Illustrative output.) `history` (previous turns) can be sent so follow-ups like "¿y en negro?" work.

## Limits

The data has **no prices, stock or store policies**. Questions about them, and anything that is not product search, get a fixed refusal with `products: []`. If nothing in the catalog matches, the answer is a fixed "no match" message, also with `products: []`.
