"""Streamlit chat interface for the Energy Price Planning Agent (live SMARD prices).

Run from the project root:
    pip install -r requirements.txt
    streamlit run app.py

NVIDIA NIM needs NVIDIA_API_KEY in the project-root .env; local Qwen needs a running llama-server.
"""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import streamlit as st

from energy_agent.agent import Agent
from energy_agent.sources import SmardPriceSource
from energy_agent.tools import EnergyTools
from main import build_llm  # importing main also loads the project-root .env

EXAMPLE_QUESTION = """For 16 November 2026, I have a 2 kW washing machine that needs to operate for exactly three consecutive hours.

1. Retrieve all available hourly electricity prices for that date.
2. Calculate the average, minimum and maximum electricity prices for the day.
3. Find the cheapest consecutive three-hour window between 07:00 and 22:00.
4. Calculate the wholesale operating cost for the recommended window.
5. Calculate the cost of running the washing machine during the fixed evening period of 18:00–21:00.
6. Compare both options and calculate the absolute and percentage savings.
7. Recommend the cheaper option and explain your decision.

Do not invent missing prices. If the required observations are incomplete, explicitly explain the limitation."""

PROVIDERS = {"nvidia": "NVIDIA NIM · Nemotron (hosted)", "llama-cpp": "Local Qwen · llama.cpp"}

st.set_page_config(page_title="Energy Price Agent", page_icon="⚡", layout="centered")


@st.cache_resource(show_spinner=False)
def get_agent(provider: str, max_steps: int) -> Agent:
    """Build the agent once per provider/limit (reuses the CLI's provider logic), always with live SMARD data."""
    llm = build_llm(SimpleNamespace(provider=provider, model=None, base_url=None))
    return Agent(llm=llm, tools=EnergyTools(price_source=SmardPriceSource()), max_steps=max_steps, verbose=True)


class LiveTrace:
    """File-like object that receives Agent.run()'s printed log.

    It keeps the full log for 'View behind the scenes' and, line by line, shows the
    agent's progress (iteration, tool called, arguments, result status) in a live status box.
    """

    def __init__(self, status) -> None:
        self.status = status
        self.log: list[str] = []
        self.tools: list[str] = []
        self._pending = ""
        self._block: list[str] | None = None  # multi-line JSON after [TOOL ARGUMENTS] / [TOOL OBSERVATION]
        self._block_kind = ""

    def write(self, text: str) -> int:
        self.log.append(text)
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._handle(line)
        return len(text)

    def flush(self) -> None:
        if self._pending:
            self._handle(self._pending)
            self._pending = ""

    def _handle(self, line: str) -> None:
        if self._block is not None:  # collecting an indented JSON block
            self._block.append(line)
            if line == "}":
                self._finish_block()
            return
        if line.startswith("============ AGENT ITERATION"):
            step = line.split("ITERATION")[1].strip(" =")
            self.status.update(label=f"Iteration {step}: the model is deciding the next step…")
        elif line.startswith("[AGENT DECISION] Call tool:"):
            name = line.split(":", 1)[1].strip()
            self.tools.append(name)
            self.status.update(label=f"Running tool `{name}`…")
            self.status.markdown(f"🔧 **Calling tool `{name}`**")
        elif line.startswith(("[TOOL ARGUMENTS] ", "[TOOL OBSERVATION] ")):
            kind, rest = line.split("] ", 1)
            self._block_kind, self._block = kind.strip("["), [rest]
            if rest.strip() != "{":  # single-line JSON
                self._finish_block()
        elif line.startswith("[AGENT] Final answer"):
            self.status.update(label="Writing the final answer…")

    def _finish_block(self) -> None:
        raw, kind = "\n".join(self._block or []), self._block_kind
        self._block, self._block_kind = None, ""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        if kind == "TOOL ARGUMENTS":
            args = ", ".join(f"{k}={v!r}" for k, v in data.items())
            self.status.caption(f"arguments: {args}")
        elif kind == "TOOL OBSERVATION" and isinstance(data, dict):
            state = data.get("status", "ok")
            if state == "tool_error":
                self.status.error(f"tool error: {data.get('error')}")
            else:
                self.status.caption(f"result received · status: {state}")

    def text(self) -> str:
        return "".join(self.log).strip()


def show_details(tools: list[str], trace: str) -> None:
    if tools:
        st.caption("Tools used: " + " → ".join(f"`{t}`" for t in tools))
    if trace:
        with st.expander("View behind the scenes"):
            st.code(trace, language="text")


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM provider", list(PROVIDERS), format_func=PROVIDERS.get)
    max_steps = st.slider("Max agent iterations", min_value=1, max_value=15, value=12)
    st.caption("Prices: live SMARD day-ahead data, DE-LU bidding zone.")
    if st.button("Clear chat"):
        st.session_state.history = []

# ----------------------------------------------------------------------------- main screen
st.title("⚡ Energy Price Planning Agent")
st.caption("Live German DE-LU day-ahead prices · every question is answered independently (no chat memory)")

if "history" not in st.session_state:
    st.session_state.history = []

with st.container(border=True):
    st.markdown("**Example question**")
    st.markdown(EXAMPLE_QUESTION)
    if st.button("Ask this question", type="primary"):
        st.session_state.pending = EXAMPLE_QUESTION

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        show_details(turn.get("tools", []), turn.get("trace", ""))

prompt = st.chat_input("Ask about prices, cheap time windows or appliance costs")
prompt = prompt or st.session_state.pop("pending", None)

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            agent = get_agent(provider, max_steps)
        except ValueError as exc:  # e.g. NVIDIA_API_KEY missing
            st.error(f"Configuration error: {exc}")
            st.stop()

        status = st.status("Starting the agent…", expanded=True)
        live = LiveTrace(status)
        error = None
        with contextlib.redirect_stdout(live):
            try:
                answer = agent.run(prompt)
            except Exception as exc:  # unreachable server, quota, malformed model output
                answer, error = None, str(exc)
        live.flush()

        if error:
            status.update(label="The agent stopped with an error", state="error", expanded=False)
            hint = (
                "Check NVIDIA_API_KEY in .env, internet access and API quota."
                if provider == "nvidia"
                else "Check that llama-server is running with the GGUF model and --jinja."
            )
            content = f"**Error:** {error}\n\n{hint}"
            st.error(content)
        else:
            n = len(live.tools)
            status.update(label=f"Done · {n} tool call{'s' if n != 1 else ''}", state="complete", expanded=False)
            content = answer
            st.markdown(content)
        show_details(live.tools, live.text())

    st.session_state.history.append({"role": "assistant", "content": content, "tools": live.tools, "trace": live.text()})
