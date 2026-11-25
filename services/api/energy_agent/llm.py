"""LLM adapters for local llama.cpp and hosted NVIDIA NIM.

Both speak the OpenAI-compatible chat-completions API. The reply is read as an SSE stream (`_stream_chat`) and
converted to the project's own types (`_normalize_message`), so the agent never sees provider details.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any, Callable
import requests


@dataclass
class ToolCall:
    """A normalized tool call returned by the model.

    one example is like this data structure:
    {
        "name": "get_hourly_prices",
        "arguments": {
            "date": "2024-06-01",
            "region": "DE-LU"
        },
        "call_id": "call_0_0"
    }
    """

    # name of the tool to call, e.g., "get_hourly_prices"
    name: str

    # arguments for the tool call, e.g., {"date": "2024-06-01", "region": "DE-LU"}
    # these are basically the parameters or arguments, the function 'get_hourly_prices' expects
    # to receive to perform a certain task
    #
    # dictionary of key-value pairs where keys are argument names and values are argument values.
    arguments: dict[str, Any]

    # unique identifier for the tool call
    call_id: str


@dataclass
class LLMResponse:
    """Provider-independent response consumed by the agent loop."""

    content: str
    tool_calls: list[ToolCall]
    raw_assistant_message: dict[str, Any]


def _normalize_message(message: dict, messages: list[dict], provider: str) -> LLMResponse:
    """Convert an OpenAI-style assistant message to project types.

    Both llama.cpp and NVIDIA NIM describe the assistant's turn as one message with `content`, optional
    `tool_calls` and, for some models, `reasoning_content`. `_stream_chat` assembles that message from the
    streamed chunks; this helper keeps provider-specific HTTP handling separate from the agent's
    provider-independent response format.
    """
    # How does such a message look like?
    # {
    #     "role": "assistant",
    #     "content": "The energy price in DE-LU on 2024-06-01 is 0.25 EUR/kWh.",
    #     "tool_calls": [
    #         {
    #             "id": "call_0_0",
    #             "type": "function",
    #             "function": {
    #                 "name": "get_hourly_prices",
    #                 "arguments": "{\"date\": \"2024-06-01\", \"region\": \"DE-LU\"}"
    #             }
    #         },
    #         {
    #             "id": "call_0_1",
    #             "type": "function",
    #             "function": {
    #                 "name": "get_hourly_prices",
    #                 "arguments": "{\"date\": \"2024-06-02\", \"region\": \"DE-LU\"}"
    #             }
    #         }
    #     ]
    # }
    content = message.get("content")
    content = content if isinstance(content, str) else ""

    # The model may return multiple tool calls in a single response (the two in the example above).
    tool_calls, raw_calls = [], []
    for index, item in enumerate(message.get("tool_calls") or []):
        if not isinstance(item, dict):
            raise ValueError(f"{provider} returned malformed tool call.")
        fn = item.get("function") or {}
        name = fn.get("name")
        if not name:
            raise ValueError(f"{provider} tool call missing function name.")
        # we need to convert the arguments from an encoded JSON string to a dictionary
        # because the server returns the arguments as a JSON string,
        # but for Argument Unpacking, we need it as a dictionary to pass to the tool function.
        raw_args = fn.get("arguments", {})
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError as exc:
            raise ValueError(f"{provider} returned invalid arguments for {name}") from exc
        if not isinstance(args, dict):
            raise ValueError(f"{provider} arguments for {name} must be an object.")
        # The tool result must carry this same ID when it is sent back to the model.
        call_id = item.get("id") or f"call_{len(messages)}_{index}"
        # we need to convert the arguments dictionary back to a JSON string because the server
        # expects the arguments to be in JSON format when they are sent back to the model.
        args_json = json.dumps(args, ensure_ascii=False)
        tool_calls.append(ToolCall(name, args, call_id))
        raw_calls.append({"id": call_id, "type": "function",
                         "function": {"name": name, "arguments": args_json}})
    # Keep standard OpenAI message fields. Nemotron may return reasoning_content;
    # preserve it so the provider receives the complete assistant turn next time.
    raw_message = {"role": "assistant", "content": content or None}
    if message.get("reasoning_content"):
        raw_message["reasoning_content"] = message["reasoning_content"]
    if raw_calls:
        raw_message["tool_calls"] = raw_calls
    return LLMResponse(content, tool_calls, raw_message)


def _stream_chat(session: requests.Session, url: str, payload: dict, timeout: float,
                 provider: str, on_text_delta: Callable[[str], None] | None = None) -> LLMResponse:
    """Read SSE chunks, concatenate partial tool arguments, and stream visible text.

    Each `data:` line is a small JSON chunk whose choices[0].delta carries one piece of the reply: some
    `content` text, some `reasoning_content`, or a fragment of a tool call (`tool_calls[index]`: the id, the
    name, and the `arguments` JSON string arriving a few characters at a time). The pieces are joined per
    index, and the finished message is handed to `_normalize_message`.
    """
    payload = {**payload, "stream": True}
    content_parts, reasoning_parts = [], []
    calls: dict[int, dict] = {}
    try:
        with session.post(url, json=payload, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                # OpenAI-compatible SSE streams are UTF-8. Explicit decoding avoids
                # requests' response-charset guess corrupting en/em dashes, €, and smart quotes.
                line = raw_line.decode("utf-8-sig") if isinstance(raw_line, bytes) else str(raw_line)
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    text_delta = delta.get("content")
                    if isinstance(text_delta, str) and text_delta:
                        content_parts.append(text_delta)
                        if on_text_delta:
                            on_text_delta(text_delta)
                    reason_delta = delta.get("reasoning_content")
                    if isinstance(reason_delta, str):
                        reasoning_parts.append(reason_delta)
                    for tc in delta.get("tool_calls") or []:
                        idx = int(tc.get("index", 0))
                        entry = calls.setdefault(idx, {"id": "", "type": "function",
                                                   "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            entry["id"] += tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            entry["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            entry["function"]["arguments"] += fn["arguments"]
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" Server response: {exc.response.text[:1000]}"
        raise RuntimeError(f"Could not call {provider} at {url}.{detail}") from exc

    message = {"role": "assistant", "content": "".join(content_parts) or None}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    return _normalize_message(message, payload.get("messages", []), provider)


class LlamaCppServerLLM:
    """Connect to a local llama.cpp server through its OpenAI-compatible API.

    For local llama.cpp deployments, start the llama-server separately with the GGUF model loaded and --jinja
    enabled. The server handles model inference; this class only sends chat messages and tool schemas, then
    normalizes the returned OpenAI-style tool calls.
    """

    def __init__(self, model="Qwen3.5-4B", base_url="http://127.0.0.1:8080/v1",
                 timeout: float = 300.0):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.chat_url, self.timeout = f"{self.base_url}/chat/completions", timeout
        self.session = requests.Session()

    def generate(self, messages, tools, on_text_delta=None,
                 temperature=None, json_mode=False) -> LLMResponse:
        payload = {"model": self.model, "messages": messages,
                   "temperature": 0 if temperature is None else temperature, "max_tokens": 1200}
        if tools:
            payload.update({
                "tools": tools,
                # "tool_choice": "auto" -->> This tells the OpenAI-compatible API that the model can decide whether to:
                #     - Generate a normal textual response.
                #     - Request one of the available tools.
                # Meaning, the model can name one or more required tools and provide the necessary arguments for
                # them, or just generate a normal textual response without calling any tool.
                "tool_choice": "auto",
                # when the LLM supports multiple tool calls in parallel, you will see it return several tool calls
                # in a single response. The agent then executes them one after another (see agent.py).
                "parallel_tool_calls": True,
            })
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            return _stream_chat(self.session, self.chat_url, payload, self.timeout,
                               "llama.cpp", on_text_delta)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} Start llama-server with the model and --jinja.") from exc


class NvidiaNIMLLM:
    """Connect to NVIDIA-hosted Nemotron through the NVIDIA NIM API.

    The API key is read from NVIDIA_API_KEY. It is sent only in the HTTP
    Authorization header and is never included in the request body or logs.
    """

    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
    DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"

    def __init__(self, model=None, base_url=None, api_key=None, timeout=300.0,
                 reasoning_effort=None, max_tokens=None):
        self.model = model or os.getenv("NEMOTRON_MODEL", self.DEFAULT_MODEL)
        self.base_url = (base_url or os.getenv("NVIDIA_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.chat_url, self.timeout = f"{self.base_url}/chat/completions", timeout
        self.reasoning_effort = reasoning_effort or os.getenv("NEMOTRON_REASONING_EFFORT", "low")
        self.max_tokens = max_tokens or int(os.getenv("NEMOTRON_MAX_TOKENS", "8192"))
        key = api_key or os.getenv("NVIDIA_API_KEY")
        if not key:
            raise ValueError("NVIDIA_API_KEY is not set. Add it to the project-root .env or environment.")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {key}", "Accept": "application/json",
                                    "Content-Type": "application/json"})

    def generate(self, messages, tools, on_text_delta=None,
                 temperature=None, json_mode=False) -> LLMResponse:
        payload = {"model": self.model, "messages": messages,
                  "temperature": 1.0 if temperature is None else temperature,
                  "top_p": 0.95, "max_tokens": self.max_tokens,
                  "reasoning_effort": self.reasoning_effort}
        if tools:
            # parallel_tool_calls False: the model asks for one tool per reply, so a multi-step question takes one
            # round (one LLM call) per tool.
            payload.update({"tools": tools, "tool_choice": "auto", "parallel_tool_calls": False})
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return _stream_chat(self.session, self.chat_url, payload, self.timeout, "NVIDIA NIM", on_text_delta)
