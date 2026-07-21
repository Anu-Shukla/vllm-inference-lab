import requests

URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

payload = {
        "model": MODEL,
        "prompt": "The main bottleneck in LLM decode is",
        "max_tokens": 32,
        "temperature": 0,
        }

response = requests.post(URL, json=payload)
response.raise_for_status()

data = response.json()

print(data["choices"][0]["text"])
print(data["usage"])


