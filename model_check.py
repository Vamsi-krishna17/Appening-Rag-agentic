import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={
        "Authorization": f"Bearer {api_key}"
    }
)

response.raise_for_status()

models = response.json()["data"]

print(f"Total models: {len(models)}\n")

for model in models:
    print(model["id"])