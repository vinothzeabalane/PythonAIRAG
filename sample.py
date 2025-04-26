import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "llama3.2:3b",
    "prompt": "Hello, world!",
    "stream": False
}
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    print(response.json())
except requests.RequestException as e:
    print(f"Error: {e}")


