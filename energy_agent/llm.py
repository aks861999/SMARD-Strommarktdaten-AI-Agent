"""OpenAI-compatible adapters with SSE streaming and normalized tool-call handling."""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any, Callable
import requests


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall]
    raw_assistant_message: dict[str, Any]


def _normalize_message(message: dict, messages: list[dict], provider: str) -> LLMResponse:
    content = message.get("content")
    content = content if isinstance(content, str) else ""
    tool_calls, raw_calls = [], []
    for index, item in enumerate(message.get("tool_calls") or []):
        if not isinstance(item, dict):
            raise ValueError(f"{provider} returned malformed tool call.")
        fn = item.get("function") or {}
        name = fn.get("name")
        if not name:
            raise ValueError(f"{provider} tool call missing function name.")
        raw_args = fn.get("arguments", {})
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError as exc:
            raise ValueError(f"{provider} returned invalid arguments for {name}") from exc
        if not isinstance(args, dict):
            raise ValueError(f"{provider} arguments for {name} must be an object.")
        call_id = item.get("id") or f"call_{len(messages)}_{index}"
        args_json = json.dumps(args, ensure_ascii=False)
        tool_calls.append(ToolCall(name, args, call_id))
        raw_calls.append({"id": call_id, "type": "function",
                         "function": {"name": name, "arguments": args_json}})
    raw_message = {"role": "assistant", "content": content or None}
    if message.get("reasoning_content"):
        raw_message["reasoning_content"] = message["reasoning_content"]
    if raw_calls:
        raw_message["tool_calls"] = raw_calls
    return LLMResponse(content, tool_calls, raw_message)


def _stream_chat(session: requests.Session, url: str, payload: dict, timeout: float,
                 provider: str, on_text_delta: Callable[[str], None] | None = None) -> LLMResponse:
    """Read SSE chunks, concatenate partial tool arguments, and stream visible text."""
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
            payload.update({"tools": tools, "tool_choice": "auto", "parallel_tool_calls": True})
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            return _stream_chat(self.session, self.chat_url, payload, self.timeout,
                               "llama.cpp", on_text_delta)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} Start llama-server with the model and --jinja.") from exc


class NvidiaNIMLLM:
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
            payload.update({"tools": tools, "tool_choice": "auto", "parallel_tool_calls": False})
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return _stream_chat(self.session, self.chat_url, payload, self.timeout, "NVIDIA NIM", on_text_delta)
