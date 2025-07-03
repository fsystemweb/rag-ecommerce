import fitz
import os
import json
import numpy as np
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key="eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDExMzI0MDAwNjg1MDgwNjQ5MDQ3MiIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwOTIwNzI1NiwidXVpZCI6ImJmNzg4N2FlLTE1NDEtNGVlZS1iOTNhLTM3YzYzNzk4Mzk4YiIsIm5hbWUiOiJyYWciLCJleHBpcmVzX2F0IjoiMjAzMC0wNy0wMlQwNzoyMDo1NiswMDAwIn0.MOOxD7gl2o-Ka6edv_zLOcBh8yYkoaEVfLHbxXDNmPM"
)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join([page.get_text("text") for page in doc])

def extract_text_from_md(md_path):
    with open(md_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text, n=1000, overlap=200):
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
    data_path = "../../data/data.md"
    text = extract_text_from_md(data_path)
    chunks = chunk_text(text)
    embeddings = create_embeddings(chunks)
    save_model_data(chunks, embeddings)
    print(f"Saved {len(chunks)} chunks and embeddings to model_data.json")
