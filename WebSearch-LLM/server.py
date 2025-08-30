import os
import time
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Avoid importing OpenAI at import-time; import lazily when needed
OpenAI = None  # type: ignore

from websearch_tool import TOOLS, _serialize_tool_calls, perform_web_search


app = FastAPI(title="OpenAI-compatible Tool Proxy")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False


def get_provider_client():
    global OpenAI  # type: ignore
    if OpenAI is None:
        from openai import OpenAI as _OpenAI
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


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    if body.stream:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "streaming not supported in this demo"}},
        )

    client = get_provider_client()

    # Merge client-provided tools with server-provided `web_search` tool (avoid duplicates by name)
    tools_by_name: Dict[str, Dict[str, Any]] = {}
    for t in (body.tools or []):
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        name = fn.get("name")
        if name:
            tools_by_name[name] = t
    for t in TOOLS:
        fn = t.get("function", {})
        name = fn.get("name")
        if name and name not in tools_by_name:
            tools_by_name[name] = t
    merged_tools = list(tools_by_name.values())

    messages: List[Dict[str, Any]] = body.messages[:]

    # If client forces a specific function, execute it server-side immediately.
    if isinstance(body.tool_choice, dict):
        tc_type = body.tool_choice.get("type")
        fn = (body.tool_choice.get("function") or {}) if isinstance(body.tool_choice, dict) else {}
        fn_name = fn.get("name") if isinstance(fn, dict) else None
        if tc_type == "function" and fn_name == "web_search":
            # Infer query from last user message if not provided
            user_query = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
            try:
                arg_str = fn.get("arguments")
                if isinstance(arg_str, str):
                    parsed_args = json.loads(arg_str or "{}")
                    user_query = parsed_args.get("query") or user_query
                    top_k = int(parsed_args.get("top_k", 5))
                else:
                    top_k = 5
            except Exception:
                top_k = 5

            # Step 1: Run the web search tool
            tool_output = perform_web_search(query=user_query, top_k=top_k)

            # Step 2: Pass results as context to the LLM for synthesis
            context_message = {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Use ONLY the following web search results to answer the user's question. Cite URLs.\n"
                    f"Web search results: {json.dumps(tool_output['results'], ensure_ascii=False, indent=2)}"
                )
            }
            user_message = {"role": "user", "content": user_query}
            synthesis_messages = [context_message, user_message]
            client = get_provider_client()
            resp = client.chat.completions.create(
                model=body.model,
                messages=synthesis_messages,
                temperature=body.temperature or 0.2,
            )
            choice = resp.choices[0]
            llm_answer = getattr(choice.message, "content", "")

            # Step 3: Extract sources from the answer
            import re
            sources = []
            # Extract markdown [n](url) citations
            sources += re.findall(r'\[\d+\]\(([^)]+)\)', llm_answer)
            # Extract URLs in parentheses
            sources += re.findall(r'\((https?://[^)]+)\)', llm_answer)
            # Extract plain URLs
            sources += re.findall(r'(https?://[\w\.-]+(?:/[\w\./\-\?=&%]*)?)', llm_answer)
            # Remove duplicates
            sources = list(dict.fromkeys(sources))
            answer_text = llm_answer.split("Sources:")[0].strip() if "Sources:" in llm_answer else llm_answer

            created_ts = int(time.time())
            return JSONResponse(
                status_code=200,
                content={
                    "id": f"chatcmpl-{created_ts}",
                    "object": "chat.completion",
                    "created": created_ts,
                    "model": body.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": answer_text,
                                "sources": sources,
                                "web_search_results": tool_output["results"],
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

    # Perform tool-call loop server-side
    for _ in range(6):
        try:
            resp = client.chat.completions.create(
                model=body.model,
                messages=messages,
                tools=merged_tools,
                tool_choice=body.tool_choice or "auto",
                temperature=body.temperature or 0.2,
            )
        except Exception:
            # Provider unavailable -> direct search fallback
            fallback_query = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
            tool_output = perform_web_search(query=fallback_query, top_k=5)
            created_ts = int(time.time())
            return JSONResponse(
                status_code=200,
                content={
                    "id": f"chatcmpl-{created_ts}",
                    "object": "chat.completion",
                    "created": created_ts,
                    "model": body.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(tool_output, ensure_ascii=False),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        choice = resp.choices[0]
        msg = choice.message
        client = get_provider_client()

        # Merge client-provided tools with server-provided `web_search` tool (avoid duplicates by name)
        tools_by_name: Dict[str, Dict[str, Any]] = {}
        for t in (body.tools or []):
            fn = t.get("function", {}) if isinstance(t, dict) else {}
            name = fn.get("name")
            if name:
                tools_by_name[name] = t
        for t in TOOLS:
            fn = t.get("function", {})
            name = fn.get("name")
            if name and name not in tools_by_name:
                tools_by_name[name] = t
        merged_tools = list(tools_by_name.values())

        messages: List[Dict[str, Any]] = body.messages[:]

        # If client forces a specific function, execute it server-side immediately.
        if isinstance(body.tool_choice, dict):
            tc_type = body.tool_choice.get("type")
            fn = (body.tool_choice.get("function") or {}) if isinstance(body.tool_choice, dict) else {}
            fn_name = fn.get("name") if isinstance(fn, dict) else None
            if tc_type == "function" and fn_name == "web_search":
                # Infer query from last user message if not provided
                user_query = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
                try:
                    # If client provided arguments in tool_choice (non-standard), parse them
                    arg_str = fn.get("arguments")
                    if isinstance(arg_str, str):
                        pass
                    else:
                        pass
                except Exception:
                    top_k = 5

                tool_call_id = "call_web_search_1"
                # Append an assistant tool call and the tool result per OpenAI schema
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": json.dumps({"query": user_query, "top_k": top_k}),
                                },
                            }
                        ],
                    }
                )

                tool_output = perform_web_search(query=user_query, top_k=top_k)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": "web_search",
                        "content": json.dumps(tool_output, ensure_ascii=False),
                    }
                )

                # Always return synthesized answer from tool results, not model output
                from websearch_tool import _format_search_results_as_answer
                created_ts = int(time.time())
                return JSONResponse(
                    status_code=200,
                    content={
                        "id": f"chatcmpl-{created_ts}",
                        "object": "chat.completion",
                        "created": created_ts,
                        "model": body.model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": _format_search_results_as_answer(user_query, tool_output),
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )


