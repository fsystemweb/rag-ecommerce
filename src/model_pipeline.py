import fitz
import os
import json
from src.config.openai_client import client
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(pdf_path): # Alternativa pero no utilizado para nuestra solución
    doc = fitz.open(pdf_path)
    return "".join([page.get_text("text") for page in doc])

def extract_text_from_md(md_path):
    with open(md_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text, n=1000, overlap=200):
    return [text[i:i+n] for i in range(0, len(text), n - overlap)]

def create_embeddings(chunks, model="BAAI/bge-multilingual-gemma2"):
    response = client.embeddings.create(model=model, input=chunks)
    embeddings = [item.embedding for item in response.data]
    return embeddings

def save_model_data(chunks, embeddings, output_file="model_data.json"):
    data = [{"chunk": c, "embedding": e} for c, e in zip(chunks, embeddings)]
    with open(output_file, "w") as f:
        json.dump(data, f)

def generate_app_embeddings(text: str) -> None:
    if not text.strip():
        raise ValueError("Input text is empty.")
    try:
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("Text chunking failed: no chunks created.")

        model_name = os.getenv("EMBEDDINGS_MODEL")
        if not model_name:
            raise EnvironmentError("Missing EMBEDDINGS_MODEL environment variable.")

        embeddings = create_embeddings(chunks, model_name)
        save_model_data(chunks, embeddings)
    except Exception as e:
        print(f"Embedding error: {e}")
        raise

if __name__ == "__main__":
    data_path = "../data/data.md"
    text = extract_text_from_md(data_path)
    chunks = chunk_text(text)
    embeddings = create_embeddings(chunks, os.getenv("EMBEDDINGS_MODEL"))
    save_model_data(chunks, embeddings)
    print(f"Saved {len(chunks)} chunks and embeddings to model_data.json")
