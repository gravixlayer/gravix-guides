import requests
import json

url = "http://127.0.0.1:8000/v1/chat/completions"
payload = {
    "model": "microsoft/phi-4",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant with a web_search tool. Cite URLs."},
        {"role": "user", "content": "What is Gravix Layer?"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web and return top results with title, url and snippet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        }
    ],
    "tool_choice": {"type": "function", "function": {"name": "web_search"}},
    "temperature": 0.2
}

response = requests.post(url, json=payload)
print(f"Status Code: {response.status_code}")
print(f"Response Content: {response.text}")
try:
    response_json = response.json()
    print(json.dumps(response_json, indent=4))
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
