"""LLM adapters for local llama.cpp and hosted NVIDIA NIM."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class ToolCall:
    """A normalized tool call returned by the model.
    
    one example is like this data structure:
    {
        "name": "get_energy_price",
        "arguments": {
            "date": "2024-06-01",
            "region": "DE-LU"
        },
        "call_id": "call_0_0"
    }

    """

    # name of the tool to call, e.g., "get_energy_price"
    name: str



    '''
    arguments for the tool call, e.g., {"date": "2024-06-01", "region": "DE-LU"}
    these are basically the parameters or arguments, the function 'get_energy_price' expects 
    to receive to perform a certain task

    dictionary of key-value pairs where keys are argument names and values are argument values.

    '''
    arguments: dict[str, Any]


    # unique identifier for the tool call                  
    call_id: str


@dataclass
class LLMResponse:
    """Provider-independent response consumed by the agent loop.
    
    For local llama.cpp deployments, start the llama-server separately with the GGUF model loaded and --jinja enabled.
        
    The server handles model inference; this class only sends chat messages and
    tool schemas, then normalizes the returned OpenAI-style tool calls.
    
    """

    content: str
    tool_calls: list[ToolCall]
    raw_assistant_message: dict[str, Any]


def _normalize_chat_response(
    response: requests.Response,
    messages: list[dict[str, Any]],
    provider_name: str,
) -> LLMResponse:
    """Convert an OpenAI-compatible chat-completion response to project types.

    Both llama.cpp and NVIDIA NIM return the assistant message inside
    choices[0].message. This helper keeps provider-specific HTTP handling
    separate from the agent's provider-independent response format.
    """
    try:
        result = response.json()
        message = result["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"{provider_name} returned an unexpected chat-completion response."
        ) from exc



    '''
    How does a message content look like?
    {
        "role": "assistant",
        "content": "The energy price in DE-LU on 2024-06-01 is 0.25 EUR/kWh.",
        "tool_calls": [
            {
                "id": "call_0_0",
                "type": "function",
                "function": {
                    "name": "get_energy_price",
                    "arguments": "{\"date\": \"2024-06-01\", \"region\": \"DE-LU\"}"
                }
            },
            {
                "id": "call_0_1",
                "type": "function",
                "function": {
                    "name": "get_energy_price",
                    "arguments": "{\"date\": \"2024-06-02\", \"region\": \"DE-LU\"}"
                }
            }
        ]
    }
    '''

    content_value = message.get("content")
    content = content_value if isinstance(content_value, str) else ""
    returned_tool_calls = message.get("tool_calls") or []


    '''
    The model may return multiple tool calls in a single response.
    example of returned_tool_calls:
    [
        {
            "id": "call_0_0",
            "type": "function",
            "function": {
                "name": "get_energy_price",
                "arguments": "{\"date\": \"2024-06-01\", \"region\": \"DE-LU\"}"
            }
        },
        {
            "id": "call_0_1",
            "type": "function",
            "function": {
                "name": "get_energy_price",
                "arguments": "{\"date\": \"2024-06-02\", \"region\": \"DE-LU\"}"
            }
        }
    ]
    '''
    

    tool_calls: list[ToolCall] = []
    normalized_calls: list[dict[str, Any]] = []

    for index, returned_tool_call in enumerate(returned_tool_calls):
        if not isinstance(returned_tool_call, dict):
            raise ValueError(f"{provider_name} returned a malformed tool call.")

        function = returned_tool_call.get("function") or {}
        name = function.get("name")
        if not name:
            raise ValueError("The model returned a tool call without a function name.")


        # we need to convert the arguments from an encoded JSON string to a dictionary 
        # because the llama.cpp server returns the arguments as a JSON string, 
        # but for Argument Unpacking, we need it as a dictionary to pass to the tool function.
        raw_arguments = function.get("arguments", {})
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"The model returned invalid JSON arguments for tool '{name}'."
                ) from exc
        else:
            arguments = raw_arguments

        if not isinstance(arguments, dict):
            raise ValueError(f"The model returned non-object arguments for tool '{name}'.")

        # The tool result must carry this same ID when it is sent back to the model.
        call_id = returned_tool_call.get("id") or f"call_{len(messages)}_{index}"



        # we need to convert the arguments dictionary to a JSON string because the llama.cpp server
        # expects the arguments to be in JSON format when sending back to the model.
        arguments_json = json.dumps(arguments, ensure_ascii=False)

        tool_calls.append(ToolCall(name=name, arguments=arguments, call_id=call_id))
        normalized_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments_json},
            }
        )

    # Keep standard OpenAI message fields. Nemotron may return reasoning_content;
    # preserve it so the provider receives the complete assistant turn next time.
    raw_assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": content if content else None,
    }
    reasoning_content = message.get("reasoning_content")
    if reasoning_content:
        raw_assistant_message["reasoning_content"] = reasoning_content

    
    if normalized_calls:
        raw_assistant_message["tool_calls"] = normalized_calls

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        raw_assistant_message=raw_assistant_message,
    )


class LlamaCppServerLLM:
    """Connect to a local llama.cpp server through its OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "Qwen3.5-4B",
        base_url: str = "http://127.0.0.1:8080/v1",
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/chat/completions"
        self.timeout = timeout
        self.session = requests.Session()

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,


            '''
            "tool_choice": "auto" -->> This tells the OpenAI-compatible API that the model can decide whether to:
                - Generate a normal textual response.
                - Request one of the available tools.

            Meaning, the model can name a/or more than one required tools and provide the necessary arguments for that tool. 
            or just generate a normal textual response without calling any tool.
            '''

            "tool_choice": "auto",
            "parallel_tool_calls": True, # when LLM Agent supports multiple tool calls in parallel, then you will see the LLM is returning multiple tool calls in a single response, and the agent will execute them in parallel.
            "temperature": 0,
            "max_tokens": 1200,
        }

        try:
            response = self.session.post(
                self.chat_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = ""
            if getattr(exc, "response", None) is not None:
                detail = f" Server response: {exc.response.text[:1000]}"
            raise RuntimeError(
                "Could not reach the local llama.cpp server at "
                f"{self.chat_url}. Start llama-server with your GGUF model and "
                f"the --jinja flag.{detail}"
            ) from exc

        return _normalize_chat_response(response, messages, "llama.cpp")


class NvidiaNIMLLM:
    """Connect to NVIDIA-hosted Nemotron through the NVIDIA NIM API.

    The API key is read from NVIDIA_API_KEY. It is sent only in the HTTP
    Authorization header and is never included in the request body or logs.
    """

    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
    DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model or os.getenv("NEMOTRON_MODEL", self.DEFAULT_MODEL)
        self.base_url = (
            base_url or os.getenv("NVIDIA_BASE_URL", self.DEFAULT_BASE_URL)
        ).rstrip("/")
        self.chat_url = f"{self.base_url}/chat/completions"
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort or os.getenv(
            "NEMOTRON_REASONING_EFFORT", "low"
        )
        self.max_tokens = max_tokens or int(os.getenv("NEMOTRON_MAX_TOKENS", "8192"))

        key = api_key or os.getenv("NVIDIA_API_KEY")
        if not key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. Add it to the project-root .env file "
                "or set it as an environment variable."
            )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "stream": False,
        }

        try:
            response = self.session.post(
                self.chat_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = ""
            if getattr(exc, "response", None) is not None:
                detail = f" Server response: {exc.response.text[:1000]}"
            raise RuntimeError(
                "Could not call NVIDIA NIM at "
                f"{self.chat_url}. Check internet access, NVIDIA_API_KEY, "
                f"model availability and API quota.{detail}"
            ) from exc

        return _normalize_chat_response(response, messages, "NVIDIA NIM")
