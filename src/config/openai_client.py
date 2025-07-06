
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("NEBIUS_API"),
    api_key=os.getenv("NEBIUS_API_KEY")
)
