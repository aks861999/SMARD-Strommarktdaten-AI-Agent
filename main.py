"""CLI entry point for the Energy Price Planning Agent."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from energy_agent.agent import Agent
from energy_agent.llm import LlamaCppServerLLM, NvidiaNIMLLM
from energy_agent.sources import DemoPriceSource, SmardPriceSource
from energy_agent.tools import EnergyTools


# Load the project-root .env without overriding variables already set by the OS.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A small tool-using German electricity price agent."
    )
    parser.add_argument(
        "--provider",
        choices=["llama-cpp", "nvidia"],
        default="llama-cpp",
        help="LLM provider: local Qwen through llama.cpp, or hosted NVIDIA NIM.",
    )
    parser.add_argument(
        "--data",
        choices=["demo", "smard"],
        default="demo",
        help="Use synthetic sample prices or fetch prices from SMARD.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override. Defaults depend on the selected provider.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible API base URL override.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.getenv("AGENT_MAX_STEPS", "0")),
        help="Optional maximum LLM/tool iterations; 0 disables the count limit (runtime limit still applies).",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=int(os.getenv("AGENT_MAX_RUNTIME_SECONDS", "900")),
        help="Runtime safety limit in seconds; 0 disables the runtime limit.",
    )
    return parser.parse_args()


def build_llm(args: argparse.Namespace):
    """Create the selected LLM adapter while keeping the same Agent interface."""
    if args.provider == "nvidia":
        model = args.model or os.getenv(
            "NEMOTRON_MODEL", NvidiaNIMLLM.DEFAULT_MODEL
        )
        base_url = args.base_url or os.getenv(
            "NVIDIA_BASE_URL", NvidiaNIMLLM.DEFAULT_BASE_URL
        )
        return NvidiaNIMLLM(model=model, base_url=base_url)

    model = args.model or "Qwen3.5-4B"
    base_url = args.base_url or "http://127.0.0.1:8080/v1"
    return LlamaCppServerLLM(model=model, base_url=base_url)


def main() -> int:
    args = parse_args()

    try:
        llm = build_llm(args)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    price_source = DemoPriceSource() if args.data == "demo" else SmardPriceSource()
    tools = EnergyTools(price_source=price_source)
    agent = Agent(
        llm=llm, tools=tools,
        max_steps=args.max_steps or None,
        max_runtime_seconds=args.max_runtime_seconds or None,
        verbose=True,
    )

    print("\nENERGY PRICE PLANNING AGENT")
    if args.provider == "nvidia":
        print(f"LLM: {llm.model} (NVIDIA NIM hosted API)")
        print(f"NVIDIA NIM API: {llm.base_url}")
    else:
        print(f"LLM: {llm.model} (served by local llama.cpp)")
        print(f"llama.cpp API: {llm.base_url}")
    print(f"Data source: {args.data}")
    if args.data == "demo":
        print("NOTICE: demo prices are synthetic; they are not live market observations.")
    print("Type 'exit' to quit.\n")

    carried_state: dict | None = None
    while True:
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return 0
        if not user_query:
            continue

        try:
            answer, a2ui_messages = agent.run(user_query, prior_state=carried_state)
            carried_state = agent.last_turn_state
            print(f"\nAgent: {answer}\n")
            if a2ui_messages:
                import json
                print(f"[A2UI] {len(a2ui_messages)} validated UI message(s):")
                print(json.dumps(a2ui_messages, indent=2, ensure_ascii=False))
        except Exception as exc:
            print(f"\nApplication error: {exc}", file=sys.stderr)
            if args.provider == "nvidia":
                print(
                    "Check the project-root .env, NVIDIA_API_KEY, internet access, "
                    "model availability and API quota.",
                    file=sys.stderr,
                )
            else:
                print(
                    "Check that llama-server is running with the GGUF loaded "
                    "and --jinja enabled.",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    raise SystemExit(main())
