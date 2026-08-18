import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": "In one sentence, explain what a data analyst does.",
        "stream": False
    }
)

print(response.json()["response"])