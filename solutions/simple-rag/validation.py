
import json
import numpy as np
from openai import OpenAI
from util.generate_response import generate_response

def evaluate_response(query, ai_response, ideal_answer):
    eval_prompt = (
        f"User Query: {query}\nAI Response:\n{ai_response}\n"
        f"True Response: {ideal_answer}\n"
        "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. "
        "Assign 1 for correct, 0.5 for partial, and 0 for incorrect."
    )
    return generate_response(eval_prompt, eval_prompt)


def validation():
    #with open("data/val.json") as f:
    #    val_data = json.load(f)

    #query = val_data[0]["question"]
    #ideal_answer = val_data[0]["ideal_answer"]

    #score = evaluate_response(query, ai_answer, ideal_answer)
    #print("Evaluation Score:\n", score)
