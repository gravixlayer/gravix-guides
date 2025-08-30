
# WebSearch-LLM

**Using Web Search with LLMs:**

WebSearch-LLM is an OpenAI-compatible proxy server that integrates a web search tool directly into the LLM workflow. This project demonstrates how LLMs can move beyond static training data and become more practical by combining their reasoning ability with real-time knowledge.

---

## Table of Contents

- [Introduction](#introduction)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Usage Example](#usage-example)
- [Extending & Customizing](#extending--customizing)
- [FAQ](#faq)
- [License](#license)

---

## Introduction

**WebSearch-LLM** lets any LLM fetch live information from the web using DuckDuckGo. Results are returned in a structured format (title, URL, snippet) that the model can reason over and cite. It supports both tool-calling models and those without native tool support, making it universally usable. Because it mirrors the OpenAI Chat Completions API, it can be plugged into existing applications without changes.

---

## How It Works

- The server exposes an OpenAI-compatible `/v1/chat/completions` endpoint.
- When a request is received, it merges any client-provided tools with its own built-in `web_search` tool.
- If the model or client requests a web search, the server performs a DuckDuckGo search and returns the top results (title, URL, snippet).
- The results are injected into the LLM's context, allowing the model to synthesize answers using real-time information and cite sources.
- If the model does not support tool-calling, the server can still perform the search and return results directly.

---

## Architecture

- **`server.py`**: FastAPI server that handles requests, manages tool calls, and orchestrates the workflow between the LLM and the web search tool.
- **`websearch_tool.py`**: Implements the DuckDuckGo search logic, result formatting, and fallback strategies (Wikipedia, DuckDuckGo Instant Answer).
- **`test.py`**: Example client script showing how to interact with the server using the OpenAI Chat Completions API format.

### Data Flow
1. **Client Request**: Sends a chat completion request (with or without tool-calling).
2. **Tool Merge**: Server merges client and built-in tools.
3. **Web Search Execution**: If requested, server performs a DuckDuckGo search.
4. **LLM Synthesis**: Results are injected into the LLM's context for answer synthesis.
5. **Response**: Server returns the answer, sources, and search results in OpenAI-compatible format.

---

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gravixlayer/gravix-guides.git
   cd gravix-guides/WebSearch-LLM
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

Get your `GRAVIXLAYER_API_KEY` from [platform.gravixlayer.com](https://platform.gravixlayer.com) and set it as an environment variable:
```bash
export GRAVIXLAYER_API_KEY=your_api_key_here
```

---

## Usage Example

Start the server using Uvicorn:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Send a request using the OpenAI Chat Completions API format (see `test.py`):

```python
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
print(json.dumps(response.json(), indent=4))
```

---

## Extending & Customizing

- **Add More Tools**: You can extend the server to support additional tools by modifying the `TOOLS` list and implementing new tool logic in `websearch_tool.py`.
- **Change Search Provider**: Swap out DuckDuckGo for another provider by updating the search logic in `websearch_tool.py`.
- **Model Choice**: The server is compatible with any OpenAI-style LLM endpoint.

---

## FAQ

**Q: Does this work with any LLM?**  
A: Yes, as long as the model supports the OpenAI Chat Completions API format. Tool-calling is supported but not required.

**Q: How do I get an API key?**  
A: Sign up at [platform.gravixlayer.com](https://platform.gravixlayer.com) and copy your API key.

**Q: Can I use this in production?**  
A: This is a demo and reference implementation. For production, review security, error handling, and scalability.

---


## License

This project is licensed under the [MIT License](LICENSE).

---

For more details, see the code and documentation in this repository.
