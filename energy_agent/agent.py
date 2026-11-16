"""The LLM-driven observe -> decide -> act loop."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from .llm import LLMResponse
from .tools import EnergyTools, TOOL_SCHEMAS


SYSTEM_PROMPT = """You are an Energy Price Planning Agent for the German DE-LU day-ahead electricity market.

Your job is to answer questions about hourly day-ahead prices, price patterns, cheap operating windows,
and estimated wholesale energy costs for electrical appliances.

You have exactly four tools available. Decide dynamically which tool or tools are needed. Do not call
every tool automatically. You may call tools in multiple iterations, using previous observations to
decide the next action. You may also answer directly when no market data is needed.

Rules:
- For factual claims about actual electricity prices, call get_hourly_prices or an appropriate analysis tool.
- Never invent, estimate, or fill missing market observations.
- Convert relative dates such as 'tomorrow' into YYYY-MM-DD using the current Berlin date supplied below.
- Use the DE-LU region unless the user explicitly asks otherwise; this project supports DE-LU only.
- Prices returned by price tools are in EUR/MWh.
- For appliance cost, use estimate_run_cost. It calculates only the wholesale energy component.
- If a tool reports missing or partial data, explain that limitation; do not silently replace it with invented values.
- Ask one concise clarification question if a required input such as appliance power or run duration is missing.
- Stay within the electricity-price domain. Be explicit when a request is outside your scope.
- Do not claim that an appliance was switched on or that a booking/action was performed.

Current Berlin date: {today}
Current Berlin date and time: {now}
"""


class Agent:
    def __init__(self, llm: Any, tools: EnergyTools, max_steps: int = 8, verbose: bool = True) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps # number of iterations the agent will perform before stopping, number of LLM call.
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    @staticmethod
    def _json_safe(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)

    def run(self, user_query: str) -> str:
        now = datetime.now(ZoneInfo("Europe/Berlin"))
        system_prompt = SYSTEM_PROMPT.format(today=now.date().isoformat(), now=now.isoformat(timespec="minutes"))

        # 'messages' is a list of messages that will be sent to the LLM. Each message is a dictionary with a 'role' and 'content' and 
        # first time the agent runs, at first the 'messages' list will have the system prompt and the user query
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        

        for step in range(1, self.max_steps + 1):
            self._log(f"\n{'=' * 12} AGENT ITERATION {step} {'=' * 12}")
            response: LLMResponse = self.llm.generate(messages=messages, tools=TOOL_SCHEMAS)
            

            self._log(f"\n[AGENT DECISION] LLM response:\n{self._json_safe(response)}")

            if response.raw_assistant_message:
                '''
                The assistant's message may contain content and tool calls. 
                We append it to the messages list for the next iteration.
                '''
                messages.append(response.raw_assistant_message)


            # if the model did not return any tool calls, we assume it has generated a final answer and we return it.
            if not response.tool_calls:
                answer = response.content.strip() or "The model returned an empty response."
                self._log("\n[AGENT] Final answer generated.")
                return answer

            '''
            
            there can be multiple tool calls in a single assistant message. For each tool call, we execute the corresponding tool function and append the observation to the messages list for the next iteration.
            '''
            for call in response.tool_calls:
                self._log(f"\n[AGENT DECISION] Call tool: {call.name}")
                self._log(f"[TOOL ARGUMENTS] {self._json_safe(call.arguments)}")
                try:
                    result = self._execute_tool(call.name, call.arguments)
                except Exception as exc:  # Send controlled tool errors back as observations.
                    result = {"status": "tool_error", "tool": call.name, "error": str(exc)}

                self._log(f"[TOOL OBSERVATION] {self._json_safe(result)}")
                tool_message = {
                    "role": "tool",
                    "content": self._json_safe(result),
                }
                # The observation must reference the exact assistant tool call.
                tool_message["tool_call_id"] = call.call_id

                # upto this point, every tool call and its observation is appended to the messages list, so that the next iteration of the agent can use it to decide the next action.
                messages.append(tool_message)

        return (
            f"I stopped because the agent reached its maximum of {self.max_steps} iterations. "
            "Please narrow the request or try again."
        )

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
