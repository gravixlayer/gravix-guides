import os
import json
import argparse
from typing import List, Dict, Any, Optional

import requests
try:
    # Prefer robust search results via duckduckgo-search package
    from duckduckgo_search import DDGS  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    DDGS = None  # type: ignore
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore
# Note: Avoid importing OpenAI at module import time to prevent hard dependency
OpenAI = None  # will be imported lazily inside chat_with_websearch


MODEL = "llama3.1:8b"


# Tool schema definition (OpenAI tools/function-calling)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return top results with title, url and snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (1-10).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    }
]


def perform_web_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    # DuckDuckGo HTML scraping for web search (no API key required)
    try:
        url = "https://html.duckduckgo.com/html/"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.post(url, data={"q": query}, headers=headers, timeout=10)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        seen_snippets = set()
        for a in soup.select(".result__a")[:max(50, top_k * 2)]:
            title = a.get_text(strip=True)
            link = a["href"]
            body_div = a.find_parent("div", class_="result__body")
            snippet = ""
            if body_div:
                # Remove extra whitespace and boilerplate
                snippet = body_div.get_text(" ", strip=True)
                snippet = " ".join(snippet.split())
                # Remove repeated title and URL from snippet
                if title in snippet:
                    snippet = snippet.replace(title, "").strip()
                if link in snippet:
                    snippet = snippet.replace(link, "").strip()
                # Remove leading/trailing domain names
                import re
                snippet = re.sub(r"^\s*([\w.-]+\.[a-z]{2,})\s*", "", snippet)
            # Only add unique, meaningful snippets
            norm_snippet = snippet.lower().strip()
            if norm_snippet and norm_snippet not in seen_snippets:
                seen_snippets.add(norm_snippet)
                results.append({
                    "title": title,
                    "url": link,
                    "snippet": snippet
                })
            if len(results) >= top_k:
                break
        if results:
            return {"query": query, "results": results, "provider": "duckduckgo_html_scrape"}
    except Exception as exc:
        print(f"[DuckDuckGo HTML Scrape] Error: {exc}")

# def perform_web_search_tavily(query: str, top_k: int = 5) -> Dict[str, Any]:
#     """Tavily Python SDK for web search (up to 50 results) -- for future use."""
#     tavily_key = os.environ.get("TAVILY_API_KEY")
#     if tavily_key:
#         try:
#             # ... (rest of the original perform_web_search_tavily content, commented out)
#             pass
#         except Exception as exc:
#             print(f"[Tavily] Error: {exc}")
#     return {"query": query, "results": [], "provider": "tavily"}

    # Tertiary: Wikipedia OpenSearch API as a broad, reliable public source
    try:
        wiki_url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "opensearch", "search": query, "limit": str(top_k), "namespace": "0", "format": "json"}
        resp = requests.get(wiki_url, params=params, timeout=10, headers={"User-Agent": "websearch-tool/1.0"})
        resp.raise_for_status()
        data = resp.json()
        titles = data[1] if len(data) > 1 else []
        descs = data[2] if len(data) > 2 else []
        urls = data[3] if len(data) > 3 else []
        results: List[Dict[str, str]] = []
        for t, d, u in zip(titles, descs, urls):
            if u:
                results.append({"title": t or "Wikipedia", "url": u, "snippet": d or ""})
        if results:
            return {"query": query, "results": results[:top_k], "provider": "wikipedia_opensearch"}
    except Exception:
        pass

    # Fallback: DuckDuckGo Instant Answer API
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_redirect": "1", "no_html": "1"}
    try:
        resp = requests.get(url, params=params, timeout=10, headers={"User-Agent": "websearch-tool/1.0"})
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"query": query, "results": [], "provider": "duckduckgo_instant_answer", "error": f"{type(exc).__name__}: {exc}"}

    results: List[Dict[str, str]] = []
    abstract_text = data.get("AbstractText")
    abstract_url = data.get("AbstractURL")
    abstract_source = data.get("AbstractSource")
    if abstract_text and abstract_url:
        results.append({"title": abstract_source or "Result", "url": abstract_url, "snippet": abstract_text})
    related = data.get("RelatedTopics", []) or []
    for item in related:
        if isinstance(item, dict) and "FirstURL" in item and "Text" in item:
            results.append({"title": item.get("Text", "")[:200], "url": item.get("FirstURL", ""), "snippet": item.get("Text", "")[:500]})
        elif isinstance(item, dict) and "Topics" in item:
            for sub in item.get("Topics", []):
                if isinstance(sub, dict) and "FirstURL" in sub and "Text" in sub:
                    results.append({"title": sub.get("Text", "")[:200], "url": sub.get("FirstURL", ""), "snippet": sub.get("Text", "")[:500]})
    return {"query": query, "results": results[:top_k], "provider": "duckduckgo_instant_answer"}


def _serialize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """Convert SDK tool call objects into plain dicts suitable for the API payload."""
    serialized: List[Dict[str, Any]] = []
    for tc in tool_calls:
        function = getattr(tc, "function", None)
        serialized.append(
            {
                "id": getattr(tc, "id", None),
                "type": "function",
                "function": {
                    "name": getattr(function, "name", "") if function else "",
                    "arguments": getattr(function, "arguments", "{}") if function else "{}",
                },
            }
        )
    return serialized


def _format_search_results_as_answer(query: str, tool_payload: Dict[str, Any]) -> str:
    results = tool_payload.get("results", []) or []
    if not results:
        return (
            f"I couldn't find results for: {query}. Try refining the query or using more specific terms."
        )

    # Synthesize a single response from all unique snippets
    snippets = []
    citations = []
    for idx, r in enumerate(results, start=1):
        snippet = (r.get("snippet") or "").strip()
        url = (r.get("url") or "").strip()
        if snippet and url:
            snippets.append(f"{snippet} [{url}]")
            citations.append(f"[{idx}]({url})")
    if not snippets:
        return f"I couldn't find results for: {query}. Try refining the query or using more specific terms."
    # Join all snippets into a single paragraph
    summary = f"Based on web search results for '{query}', here is a synthesized answer:\n\n"
    summary += " ".join(snippets)
    summary += f"\n\nSources: {', '.join(citations)}"
    return summary


def chat_with_websearch(user_prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Runs a tool-enabled chat where the model can call `web_search` as needed.
    """
    # Lazily create the inference client so imports don't require env vars
    def _get_inference_client():
        global OpenAI  # type: ignore
        if OpenAI is None:
            from openai import OpenAI as _OpenAI  # local import to avoid import-time dependency
            OpenAI = _OpenAI  # type: ignore
        api_key = os.environ.get("GRAVIXLAYER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing API key. Set GRAVIXLAYER_API_KEY (preferred) or OPENAI_API_KEY."
            )
        return OpenAI(
            base_url="https://api.gravixlayer.com/v1/inference",
            api_key=api_key,
        )

    client = _get_inference_client()
    messages: List[Dict[str, Any]] = []
    messages.append(
        {
            "role": "system",
            "content": system_prompt
            or (
                "You are a helpful assistant with web access via the `web_search` tool. "
                "When answering time-sensitive or factual questions, call `web_search`, "
                "synthesize results, and cite sources by URL."
            ),
        }
    )
    messages.append({"role": "user", "content": user_prompt})

    attempted_force_call = False
    for _ in range(4):  # up to 4 tool-call turns
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )

        choice = response.choices[0]
        msg = choice.message

        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": _serialize_tool_calls(tool_calls),
                }
            )

            for tc in tool_calls:
                if getattr(tc, "type", "") == "function":
                    fn = getattr(tc, "function", None)
                    fn_name = getattr(fn, "name", None) if fn else None
                    if fn_name == "web_search":
                        try:
                            args = json.loads(getattr(fn, "arguments", "{}") or "{}")
                        except json.JSONDecodeError:
                            args = {}

                        query = args.get("query") or ""
                        top_k = args.get("top_k", 5)

                        tool_output = perform_web_search(query=query, top_k=top_k)

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": getattr(tc, "id", None),
                                "name": "web_search",
                                "content": json.dumps(tool_output, ensure_ascii=False),
                            }
                        )

            # Continue the loop to let the model use the tool results
            continue

        # No tool call; try forcing a function call once
        if not attempted_force_call:
            attempted_force_call = True
            forced = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice={"type": "function", "function": {"name": "web_search"}},
                temperature=0.2,
            )

            forced_msg = forced.choices[0].message
            forced_tool_calls = getattr(forced_msg, "tool_calls", None) or []
            if forced_tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": forced_msg.content or "",
                        "tool_calls": _serialize_tool_calls(forced_tool_calls),
                    }
                )

                for tc in forced_tool_calls:
                    if getattr(tc, "type", "") == "function":
                        fn = getattr(tc, "function", None)
                        fn_name = getattr(fn, "name", None) if fn else None
                        if fn_name == "web_search":
                            try:
                                args = json.loads(getattr(fn, "arguments", "{}") or "{}")
                            except json.JSONDecodeError:
                                args = {}

                            query = args.get("query") or user_prompt
                            top_k = args.get("top_k", 5)

                            tool_output = perform_web_search(query=query, top_k=top_k)

                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": getattr(tc, "id", None),
                                    "name": "web_search",
                                    "content": json.dumps(tool_output, ensure_ascii=False),
                                }
                            )

                # Let the model use the tool results
                continue

        # Still no tool call; return the assistant's answer
        return msg.content or ""

    # Fallback if loop exhausted
    # As a robust fallback for providers/models without tool support, we run a search
    # ourselves and synthesize a concise answer with citations (no LLM needed).
    tool_output = perform_web_search(query=user_prompt, top_k=5)
    return _format_search_results_as_answer(user_prompt, tool_output)


def main():
    parser = argparse.ArgumentParser(description="Websearch tool-enabled chat demo")
    parser.add_argument(
        "--ask",
        type=str,
        help="Ask a question; will attempt tool-calling and fall back to direct search summary.",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Run a direct web search and print JSON results (no model involved).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return for direct search.",
    )
    args = parser.parse_args()

    if args.search:
        out = perform_web_search(args.search, top_k=args.top_k)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    question = args.ask or (
        "Find the latest macOS release and summarize key features. Include sources."
    )
    print(chat_with_websearch(question))


if __name__ == "__main__":
    main()


