"""The LLM-driven observe -> decide -> act loop.

A tool-using energy agent that emits streaming events and can add an optional dynamic chat UI.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import date, datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from .genui import build_dynamic_ui
from .llm import LLMResponse
from .tools import TOOL_SCHEMAS, EnergyTools

# Conversation memory: the last 10 question/answer exchanges (20 messages) are
# always sent. These limits are shared by the frontend, the API schema and the
# agent prompt builder. The character budgets bound prompt size so ten long
# answers cannot silently blow up the context window.
MAX_HISTORY_TURNS = 10
MAX_HISTORY_MESSAGES = MAX_HISTORY_TURNS * 2
MAX_HISTORY_CHARS = 6000          # per message
MAX_HISTORY_TOTAL_CHARS = 48000   # whole window; oldest messages are dropped first
ANSWER_CHUNK_CHARS = 160

# Slot values carried from one turn to the next so follow-ups such as
# "make it four hours instead" keep their referents.
STATE_KEYS = (
    "date",
    "region",
    "duration_hours",
    "power_kw",
    "start_hour",
    "earliest_hour",
    "latest_hour",
)

# Dynamic context (date, carried state) is appended in _system_message() by
# concatenation, not str.format. That lets this constant contain literal braces,
# such as JSON examples, without needing {{ }} escaping.
SYSTEM_PROMPT = """You are an Energy Price Planning Agent for the German DE-LU day-ahead electricity market.
You answer questions about hour-by-hour prices on a requested day, that day's price pattern,
cheap operating windows, and the wholesale energy cost of running an appliance.
Call a tool when current market data or a calculation is needed. Do not repeat identical tool calls.
Never invent or estimate missing observations. Prices are for the DE-LU bidding zone in EUR/MWh.
estimate_run_cost returns wholesale energy cost only; it excludes taxes, grid fees, supplier
margin and retail charges. State that limitation whenever you present a cost.
If a required input is genuinely missing, ask one concise question.
Stay in domain. Never claim that an appliance was switched on or that a real-world action was taken.
Respond to the last user message only. Earlier messages are background: use them to resolve
follow-ups, pronouns and omitted details, and to answer questions about the conversation itself
(for example "what was that again?"). Do not re-answer earlier questions the user did not ask again.
Earlier answers are text only: if you need a price or figure that is not stated there, call a tool.
When carried state is supplied below, use it only to fill in details of a follow-up that
omits them. If the user asks a new, unrelated question or states their own details, ignore it.
Answer in ordinary prose. Do not emit UI JSON, HTML, or code: a separate planner
may add optional inline UI after your answer, and only its validated output is rendered.
"""


class Agent:
    def __init__(self, llm: Any, tools: EnergyTools, max_steps: int | None = None,
                 max_runtime_seconds: int | None = 900, verbose: bool = False):
        self.llm, self.tools = llm, tools
        # max_steps: optional ceiling on the number of iterations (one iteration = one LLM call). None, the
        # default, means no count limit; max_runtime_seconds still bounds the run.
        self.max_steps = max_steps if max_steps and max_steps > 0 else None
        self.max_runtime_seconds = max_runtime_seconds if max_runtime_seconds and max_runtime_seconds > 0 else None
        self.verbose = verbose
        # Populated after each run so callers can carry slots into the next turn.
        self.last_turn_state: dict[str, Any] = {}

    def _emit(self, callback: Callable[[dict], None] | None, event: dict) -> None:
        if callback:
            callback(event)

    @staticmethod
    def _json_safe(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _call_key(name: str, args: dict[str, Any]) -> str:
        normalized = json.dumps({"name": name, "arguments": args}, sort_keys=True,
                               separators=(",", ":"), ensure_ascii=False, default=str)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _public_message(message: dict[str, Any]) -> dict[str, Any]:
        """Strip provider-internal reasoning from the user-facing trace."""
        return {key: value for key, value in message.items() if key != "reasoning_content"}

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        if not text:
            return []
        return [text[index:index + ANSWER_CHUNK_CHARS]
                for index in range(0, len(text), ANSWER_CHUNK_CHARS)]

    @staticmethod
    def _normalize_state(state: Any) -> dict[str, Any]:
        """Accept only known slot values and coerce them to safe scalar types."""
        if not isinstance(state, dict):
            return {}
        normalized: dict[str, Any] = {}
        for key in STATE_KEYS:
            value = state.get(key)
            if value is None:
                continue
            if key == "date":
                try:
                    normalized[key] = date.fromisoformat(str(value)).isoformat()
                except ValueError:
                    continue
            elif key == "region":
                normalized[key] = str(value)
            elif key == "power_kw":
                try:
                    normalized[key] = float(value)
                except (TypeError, ValueError):
                    continue
            else:  # whole-hour slots
                try:
                    normalized[key] = int(value)
                except (TypeError, ValueError):
                    continue
        return normalized

    def _state_block(self) -> str:
        state = self.last_turn_state
        if not state:
            return ("No previous-turn state has been carried; treat this as a fresh request "
                    "and use the current Berlin date when the user says today.")
        pairs = ", ".join(f"{key}={state[key]}" for key in STATE_KEYS if key in state)
        return ("State carried from the previous turn "
                f"(apply only to a follow-up that omits these details): {pairs}.")

    def _system_message(self, now: datetime) -> str:
        """Append dynamic context without formatting the static prompt.

        Concatenation is deliberate: SYSTEM_PROMPT is allowed to contain literal
        braces (for example JSON examples) without needing {{ }} escaping.
        """
        dynamic = (
            "Current Berlin date/time: " + now.isoformat(timespec="minutes") + "\n"
            + self._state_block()
        )
        return SYSTEM_PROMPT.rstrip() + "\n" + dynamic

    @staticmethod
    def _history_messages(conversation_context: list[dict] | None) -> list[dict[str, str]]:
        """Return the most recent valid messages that fit the history budgets.

        Walks newest to oldest so that, when the character budget is exhausted,
        it is the oldest exchanges that are dropped, never the latest ones.
        """
        kept: list[dict[str, str]] = []
        total = 0
        for item in reversed((conversation_context or [])[-MAX_HISTORY_MESSAGES:]):
            if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
                continue
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            content = content[:MAX_HISTORY_CHARS]
            if total + len(content) > MAX_HISTORY_TOTAL_CHARS:
                break
            total += len(content)
            kept.append({"role": item["role"], "content": content})
        kept.reverse()
        return kept

    def _merge_state(self, name: str, arguments: dict[str, Any], result: Any) -> None:
        """Record resolved tool arguments so the next turn can resolve follow-ups."""
        if isinstance(arguments, dict):
            for key in STATE_KEYS:
                value = arguments.get(key)
                if value is not None:
                    self.last_turn_state[key] = value
        if isinstance(result, dict):
            if isinstance(result.get("date"), str):
                self.last_turn_state.setdefault("date", result["date"])
            if name in {"find_cheapest_window", "estimate_run_cost"} and result.get("status") == "ok":
                if result.get("start_hour") is not None:
                    self.last_turn_state["start_hour"] = result["start_hour"]
        # Re-normalize after merging so downstream consumers always get clean types.
        self.last_turn_state = self._normalize_state(self.last_turn_state)

    def run(self, user_query: str, conversation_context: list[dict] | None = None,
            on_event: Callable[[dict], None] | None = None,
            on_text_delta: Callable[[str], None] | None = None,
            cancel_event: threading.Event | None = None,
            prior_state: dict[str, Any] | None = None) -> tuple[str, list[dict]]:
        now = datetime.now(ZoneInfo("Europe/Berlin"))
        self.last_turn_state = self._normalize_state(prior_state)
        # 'messages' is the list of messages that is sent to the LLM on every call. Each message is a dictionary
        # with a 'role' and 'content'. The first time through the loop it holds the system prompt, the earlier
        # conversation (at most the last 10 question/answer pairs) and the user query; the model's tool requests
        # and the tool observations are appended as the loop goes on.
        messages = [{"role": "system", "content": self._system_message(now)}]
        messages.extend(self._history_messages(conversation_context))
        messages.append({"role": "user", "content": user_query})

        observations: list[dict[str, Any]] = []
        executed: dict[str, dict[str, Any]] = {}
        deadline = (time.monotonic() + self.max_runtime_seconds
                    if self.max_runtime_seconds is not None else None)
        step = 0
        stop_reason: str | None = None
        while True:
            if self.max_steps is not None and step >= self.max_steps:
                stop_reason = f"the configured iteration safety ceiling of {self.max_steps} was reached"
                break
            if deadline is not None and time.monotonic() >= deadline:
                stop_reason = f"the configured runtime safety limit of {self.max_runtime_seconds} seconds was reached"
                break
            step += 1
            self._emit(on_event, {"type": "agent_iteration", "step": step, "max_steps": self.max_steps})
            if cancel_event is not None and cancel_event.is_set():
                self._emit(on_event, {"type": "status", "status": "Stopped by user"})
                return "Generation stopped by the user.", []
            step_label = f"step {step}/{self.max_steps}" if self.max_steps is not None else f"iteration {step}"
            self._emit(on_event, {"type": "status", "status": f"Thinking · {step_label}"})
            # Tool-capable iterations are never streamed: their text may be planning
            # text that must not reach the chat transcript.
            response: LLMResponse = self.llm.generate(messages=messages, tools=TOOL_SCHEMAS)
            if cancel_event is not None and cancel_event.is_set():
                self._emit(on_event, {"type": "status", "status": "Stopped by user"})
                return "Generation stopped by the user.", []
            tool_call_trace = [
                {"name": call.name, "arguments": call.arguments, "call_id": call.call_id}
                for call in response.tool_calls
            ]
            raw_assistant_message = response.raw_assistant_message or {
                "role": "assistant",
                "content": response.content or None,
            }
            self._emit(on_event, {
                "type": "agent_decision",
                "step": step,
                "content": response.content.strip(),
                "tool_calls": tool_call_trace,
                "raw_assistant_message": self._public_message(raw_assistant_message),
            })
            if response.raw_assistant_message:
                # The assistant's message may contain content and tool calls.
                # We append it to the messages list for the next iteration.
                messages.append(response.raw_assistant_message)

            # if the model did not return any tool calls, we assume it has generated a final answer and we return it.
            if not response.tool_calls:
                # The model's no-tool response is the answer. Generating it exactly
                # once avoids a second full answer call and the risk of the two
                # generations disagreeing.
                answer = response.content.strip() or "The model returned an empty response."
                self._emit(on_event, {
                    "type": "agent_final_response",
                    "content": answer,
                    "tool_calls": [],
                    "raw_assistant_message": self._public_message(raw_assistant_message),
                })
                self._emit(on_event, {"type": "agent_final", "status": "Final answer generated"})
                for chunk in self._chunk_text(answer):
                    if on_text_delta:
                        on_text_delta(chunk)
                    self._emit(on_event, {"type": "text_delta", "delta": chunk})
                if cancel_event is not None and cancel_event.is_set():
                    return answer, []
                self._emit(on_event, {"type": "status", "status": "Planning an optional inline UI…"})
                try:
                    ui_messages = build_dynamic_ui(self.llm, user_query, answer, observations)
                except Exception as exc:
                    # UI generation is optional: never discard a completed text answer
                    # because a separate UI-planning call failed.
                    ui_messages = []
                    self._emit(on_event, {"type": "ui_error", "error": str(exc)})
                if cancel_event is not None and cancel_event.is_set():
                    return answer, []
                for ui_message in ui_messages:
                    self._emit(on_event, {"type": "a2ui_message", "message": ui_message})
                self._emit(on_event, {"type": "turn_state", "state": self.last_turn_state})
                return answer, ui_messages

            # there can be multiple tool calls in a single assistant message. For each tool call, we execute the
            # corresponding tool function and append the observation to the messages list for the next iteration.
            for call in response.tool_calls:
                if cancel_event is not None and cancel_event.is_set():
                    self._emit(on_event, {"type": "status", "status": "Stopped by user"})
                    return "Generation stopped by the user.", []
                key = self._call_key(call.name, call.arguments)
                deduplicated = key in executed
                self._emit(on_event, {
                    "type": "tool_start", "name": call.name,
                    "arguments": call.arguments, "call_id": call.call_id,
                    "deduplicated": deduplicated,
                })
                if deduplicated:
                    result = {**executed[key], "deduplicated": True,
                              "note": "Exact call already executed during this turn; cached result reused."}
                else:
                    try:
                        result = self._execute_tool(call.name, call.arguments)
                    except Exception as exc:  # Send controlled tool errors back as observations.
                        result = {"status": "tool_error", "tool": call.name, "error": str(exc)}
                    executed[key] = result
                    # Keep the evidence set unique so the UI planner is not fed duplicates.
                    observations.append({"tool": call.name, "arguments": call.arguments, "result": result})
                    self._merge_state(call.name, call.arguments, result)
                self._emit(on_event, {
                    "type": "tool_result", "name": call.name,
                    "status": result.get("status", "received") if isinstance(result, dict) else "received",
                    "arguments": call.arguments, "result": result,
                    "deduplicated": deduplicated,
                })
                # The observation must reference the exact assistant tool call (tool_call_id). Up to this point,
                # every tool call and its observation is appended to the messages list, so that the next
                # iteration of the agent can use it to decide the next action.
                messages.append({"role": "tool", "content": self._json_safe(result),
                               "tool_call_id": call.call_id})

        answer = f"I stopped because {stop_reason or 'the agent safety budget was reached'}. Please narrow the request or increase the configured safety budget and try again."
        self._emit(on_event, {"type": "agent_limit", "reason": stop_reason})
        self._emit(on_event, {"type": "status", "status": "Preparing the safety-limit response…"})
        self._emit(on_event, {"type": "agent_final", "status": "Safety-limit response generated"})
        self._emit(on_event, {"type": "turn_state", "state": self.last_turn_state})
        return answer, []

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        function = self.tools.functions.get(name)
        if function is None:
            raise ValueError(f"Tool '{name}' is not registered. Available tools: {', '.join(self.tools.functions)}")
        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be a JSON object.")
        # Reject unknown keys rather than silently accepting model-generated arguments.
        import inspect
        allowed = set(inspect.signature(function).parameters)
        unknown = set(arguments) - allowed
        if unknown:
            raise ValueError(f"Unsupported argument(s) for {name}: {', '.join(sorted(unknown))}")
        return function(**arguments)
