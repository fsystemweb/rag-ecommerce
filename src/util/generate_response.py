from config.openai_client import client

def generate_response(query, model="meta-llama/Meta-Llama-3.1-405B-Instruct"):
    system_prompt = (
        "You are an AI assistant for sales in a ecommerce strictly answers based on the given e-commerce context. " 
        "If the answer cannot be derived from the context, respond only with: "
        "No tengo suficiente información para responder esa pregunta."
    )

    full_prompt = f"{system_prompt}\nQuestion: {query}"
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
    )
    return response.choices[0].message.content