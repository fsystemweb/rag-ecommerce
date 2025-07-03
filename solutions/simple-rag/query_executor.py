import json
import numpy as np
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key="eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDExMzI0MDAwNjg1MDgwNjQ5MDQ3MiIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwOTIwNzI1NiwidXVpZCI6ImJmNzg4N2FlLTE1NDEtNGVlZS1iOTNhLTM3YzYzNzk4Mzk4YiIsIm5hbWUiOiJyYWciLCJleHBpcmVzX2F0IjoiMjAzMC0wNy0wMlQwNzoyMDo1NiswMDAwIn0.MOOxD7gl2o-Ka6edv_zLOcBh8yYkoaEVfLHbxXDNmPM"
)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def create_query_embedding(query, model="BAAI/bge-en-icl"):
    response = client.embeddings.create(model=model, input=query)
    return response.data[0].embedding

def semantic_search(query, chunks, embeddings, k=2):
    query_vec = create_query_embedding(query)
    scores = [(i, cosine_similarity(query_vec, emb)) for i, emb in enumerate(embeddings)]
    top_indices = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    return [chunks[i] for i, _ in top_indices]

def generate_response(prompt, query, model="meta-llama/Meta-Llama-3.1-405B-Instruct"):
    full_prompt = f"{prompt}\nQuestion: {query}"
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": full_prompt}
        ]
    )
    return response.choices[0].message.content

def evaluate_response(query, ai_response, ideal_answer):
    eval_prompt = (
        f"User Query: {query}\nAI Response:\n{ai_response}\n"
        f"True Response: {ideal_answer}\n"
        "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. "
        "Assign 1 for correct, 0.5 for partial, and 0 for incorrect."
    )
    return generate_response(eval_prompt, eval_prompt)


def load_data(data_path="model_data.json"):
    """Load and prepare the model data"""
    with open(data_path) as f:
        data = json.load(f)
    chunks = [item["chunk"] for item in data]
    embeddings = [item["embedding"] for item in data]
    return chunks, embeddings

def process_query(query, chunks, embeddings):
    top_chunks = semantic_search(query, chunks, embeddings)

    system_prompt = (
        "You are an AI assistant for sales in a ecommerce strictly answers based on the given e-commerce context. " 
        "If the answer cannot be derived from the context, respond only with: "
        "No tengo suficiente información para responder esa pregunta."
    )

    context_prompt = "\n".join([
        f"Context {i + 1}:\n{chunk}\n====================================="
        for i, chunk in enumerate(top_chunks)
    ])

    ai_answer = generate_response(system_prompt, f"{context_prompt}\n\nQuestion: {query}")
    print("AI Answer:\n", ai_answer)

if __name__ == "__main__":
    chunks, embeddings = load_data()

    
    #with open("data/val.json") as f:
    #    val_data = json.load(f)

    #query = val_data[0]["question"]
    #ideal_answer = val_data[0]["ideal_answer"]


    while True:
        query = input("\nEnter your question (or 'quit' to exit): ").strip()
        
        if query.lower() == 'quit':
            break
            
        if not query:
            print("Please enter a valid question.")
            continue

        process_query(query, chunks, embeddings)






    #score = evaluate_response(query, ai_answer, ideal_answer)
    #print("Evaluation Score:\n", score)



