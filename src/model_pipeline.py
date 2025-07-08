import fitz
import os
import json
from config.openai_client import client
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join([page.get_text("text") for page in doc])

def extract_text_from_md(md_path):
    with open(md_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text, n=300, overlap=150):
    return [text[i:i+n] for i in range(0, len(text), n - overlap)]

def create_embeddings(chunks, model="BAAI/bge-en-icl"):
    response = client.embeddings.create(model=model, input=chunks)
    embeddings = [item.embedding for item in response.data]
    return embeddings

def save_model_data(chunks, embeddings, output_file="model_data.json"):
    data = [{"chunk": c, "embedding": e} for c, e in zip(chunks, embeddings)]
    with open(output_file, "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    data_path = "../data/data.md"
    text = extract_text_from_md(data_path)
    chunks = chunk_text(text)
    embeddings = create_embeddings(chunks, os.getenv("EMBEDDINGS_MODEL"))
    save_model_data(chunks, embeddings)
    print(f"Saved {len(chunks)} chunks and embeddings to model_data.json")
