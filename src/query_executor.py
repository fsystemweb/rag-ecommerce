import json

from config.openai_client import client
from util.generate_response import generate_response, cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()

def create_query_embedding(query, model="BAAI/bge-multilingual-gemma2"):
    response = client.embeddings.create(model=model, input=query)
    return response.data[0].embedding

def semantic_search(query, chunks, embeddings, k=2):
   query_vec = create_query_embedding(query)
   scores = [(i, cosine_similarity(query_vec, emb)) for i, emb in enumerate(embeddings)]
   top_indices = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
   return [chunks[i] for i, _ in top_indices]

def load_data(data_path="model_data.json"):
    with open(data_path) as f:
        data = json.load(f)
    chunks = [item["chunk"] for item in data]
    embeddings = [item["embedding"] for item in data]
    return chunks, embeddings

def build_context_prompt(top_chunks, query):
    context_prompt = "\n".join([
        f"Context {i + 1}:\n{chunk}\n====================================="
        for i, chunk in enumerate(top_chunks)
    ])
    return f"{context_prompt}\n\nQuestion: {query}"

def process_query(query, chunks, embeddings):
    top_chunks = semantic_search(query, chunks, embeddings)
    return generate_response(build_context_prompt(top_chunks, query), os.getenv("MODEL_RAG_3"))

def interactive_mode(chunks, embeddings):
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            break
        if not query:
            print("Please enter a valid question.")
            continue
        ai_answer = process_query(query, chunks, embeddings)
        print("AI Answer:\n", ai_answer)

def process_batch_query(query: str, chunks, embeddings) -> str:
    answer = process_query(query, chunks, embeddings)
    return answer

if __name__ == "__main__":
    chunks, embeddings = load_data()
    interactive_mode(chunks, embeddings)
