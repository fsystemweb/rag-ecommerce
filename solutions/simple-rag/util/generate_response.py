from config.openai_client import client


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