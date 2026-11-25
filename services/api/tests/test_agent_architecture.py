import json
from datetime import datetime, timezone
import pytest
from energy_agent.agent import Agent
from energy_agent.genui import UIValidationError, validate_ui_tree, ui_tree_to_a2ui
from energy_agent.llm import LLMResponse, ToolCall


class FakeTools:
    def __init__(self):
        self.executions = 0
        def echo(value):
            self.executions += 1
            return {"status": "ok", "value": value}
        def lookup(date, duration_hours=None):
            return {"status": "ok", "date": date, "duration_hours": duration_hours}
        self.functions = {"echo": echo, "lookup": lookup}

class FakeLLM:
    def __init__(self, sequence):
        self.sequence = iter(sequence)
        self.calls = 0
        self.messages_seen = []
    def generate(self, messages, tools, on_text_delta=None, temperature=None, json_mode=False):
        self.calls += 1
        self.messages_seen.append(list(messages))  # snapshot: the agent appends to its list later
        response = next(self.sequence)
        if on_text_delta and response.content:
            on_text_delta(response.content)
        return response

def tc(name, args, callid):
    return ToolCall(name, args, callid)

def _assistant(content):
    return LLMResponse(content, [], {"role": "assistant", "content": content})

def test_duplicate_tool_calls_execute_once_and_reuse_result():
    first = LLMResponse("", [tc("echo", {"value": "x"}, "id1"), tc("echo", {"value": "x"}, "id2")], {})
    final = _assistant("Done.")
    ui = _assistant('{"render":false}')
    fake = FakeLLM([first, final, ui])
    tools = FakeTools()
    answer, surface = Agent(fake, tools, verbose=False).run("echo this")
    assert answer == "Done."
    assert fake.calls == 3  # two agent iterations + one UI-planning call
    assert tools.executions == 1
    assert surface == []

def test_optional_ui_is_dynamically_planned_and_converted_to_surface_messages():
    plan = {
      "render": True, "title": "Interactive energy-cost estimate", "components": [
        {"id": "metric1", "type": "metric", "label": "Price", "value": "€50/MWh", "detail": "Wholesale only"},
        {"id": "chart1", "type": "chart", "title": "Hourly price", "chart_type": "line", "x_key": "hour",
         "series": [{"key": "price", "label": "Price"}], "data": [{"hour": "00:00", "price": 50}, {"hour": "01:00", "price": 30}]},
        {"id": "calc1", "type": "calculator", "title": "Change assumptions", "operation": "energy_cost",
         "output_label": "Wholesale cost", "output_unit": "€",
         "fields": [{"name": "power_kw", "label": "Power", "min": 0.1, "max": 10, "step": 0.1, "value": 2, "unit": "kW"},
                    {"name": "duration_hours", "label": "Duration", "min": 0.1, "max": 24, "step": 0.1, "value": 3, "unit": "h"},
                    {"name": "price_eur_mwh", "label": "Price", "min": 0, "max": 500, "step": 1, "value": 50, "unit": "€/MWh"}]}
      ]
    }
    messages = ui_tree_to_a2ui(plan)
    assert messages[0]["createSurface"]["catalogId"]
    assert messages[0]["version"] == "v0.9"
    components = messages[1]["updateComponents"]["components"]
    assert {c["component"] for c in components} >= {"Metric", "Chart", "Calculator", "Column"}
    assert len({c["id"] for c in components}) == len(components)

def test_validator_rejects_duplicate_ids_unsupported_types_and_bad_calculator():
    with pytest.raises(UIValidationError):
        validate_ui_tree({"components": [{"id": "a", "type": "text", "text": "one"}, {"id": "a", "type": "text", "text": "two"}]})
    with pytest.raises(UIValidationError):
        validate_ui_tree({"components": [{"id": "bad", "type": "script", "text": "alert(1)"}]})
    with pytest.raises(UIValidationError):
        validate_ui_tree({"components": [{"id": "calc", "type": "calculator", "operation": "eval", "fields": []}]})

def test_nested_card_ids_and_component_limits_are_validated():
    plan = {"components": [{"id": "card", "type": "card", "title": "Summary", "children": [{"id": "txt", "type": "text", "text": "safe text"}]}]}
    messages = ui_tree_to_a2ui(plan)
    flat = messages[1]["updateComponents"]["components"]
    ids = [x["id"] for x in flat]
    assert len(ids) == len(set(ids))
    assert "card_content" in ids
    with pytest.raises(UIValidationError):
        validate_ui_tree({"components": [{"id": f"c{i}", "type": "text", "text": "x"} for i in range(37)]})

def test_final_answer_streams_only_final_text_not_tool_iteration_text():
    first = LLMResponse("Planning…", [tc("echo", {"value": "ok"}, "call1")], {"role": "assistant", "content": "Planning…", "tool_calls": []})
    final = _assistant("This is the final answer.")
    ui = _assistant('{"render":false}')
    fake = FakeLLM([first, final, ui])
    deltas = []
    events = []
    answer, _ = Agent(fake, FakeTools(), verbose=False).run("hello", on_event=events.append, on_text_delta=deltas.append)
    assert answer == "This is the final answer."
    assert "Planning…" not in "".join(deltas)
    assert "This is the final answer." == "".join(deltas)
    assert any(e["type"] == "tool_start" for e in events)

def test_no_ui_plan_returns_empty_surface():
    fake = FakeLLM([_assistant("Plain answer."), _assistant('{"render":false}')])
    answer, messages = Agent(fake, FakeTools(), verbose=False).run("Simple question")
    assert answer == "Plain answer."
    assert messages == []
    assert fake.calls == 2  # one answer call + one UI-planning call

def test_resolved_slots_are_carried_into_follow_up_turns():
    first = LLMResponse("", [tc("lookup", {"date": "2026-11-16", "duration_hours": 3}, "c1")], {})
    agent = Agent(FakeLLM([first, _assistant("Window found."), _assistant('{"render":false}')]), FakeTools())
    agent.run("cheapest 3 hours on 2026-11-16")
    assert agent.last_turn_state == {"date": "2026-11-16", "duration_hours": 3}

    follow_llm = FakeLLM([_assistant("Updated for four hours."), _assistant('{"render":false}')])
    Agent(follow_llm, FakeTools()).run("make it four hours instead", prior_state=agent.last_turn_state)
    system_prompt = follow_llm.messages_seen[0][0]["content"]
    assert "date=2026-11-16" in system_prompt
    assert "duration_hours=3" in system_prompt

def test_system_message_uses_concatenation_not_str_format():
    agent = Agent(FakeLLM([]), FakeTools())
    agent.last_turn_state = {"date": "2026-11-16", "duration_hours": 3}
    message = agent._system_message(datetime(2026, 11, 25, 14, 3, tzinfo=timezone.utc))
    assert "Current Berlin date/time: 2026-11-25T14:03+00:00" in message
    assert "date=2026-11-16" in message
    assert "duration_hours=3" in message
    assert "{now}" not in message
    assert "{state_block}" not in message


def test_literal_braces_in_system_prompt_are_preserved(monkeypatch):
    import energy_agent.agent as agent_module
    monkeypatch.setattr(agent_module, "SYSTEM_PROMPT", 'Example JSON: {"render": false}\n')
    message = Agent(FakeLLM([]), FakeTools())._system_message(
        datetime(2026, 11, 25, 14, 3, tzinfo=timezone.utc))
    assert '{"render": false}' in message
    assert "Current Berlin date/time: 2026-11-25T14:03+00:00" in message


def test_provider_reasoning_is_not_exposed_in_user_facing_events():
    first = LLMResponse("", [tc("echo", {"value": "x"}, "c1")],
                        {"role": "assistant", "content": None, "reasoning_content": "secret chain"})
    final = LLMResponse("Answer.", [], {"role": "assistant", "content": "Answer.", "reasoning_content": "secret chain"})
    events = []
    Agent(FakeLLM([first, final, _assistant('{"render":false}')]), FakeTools()).run("q", on_event=events.append)
    assert "reasoning_content" not in json.dumps(events)

def test_ui_planner_retries_one_invalid_json_plan_then_returns_valid_surface():
    from energy_agent.genui import build_dynamic_ui
    malformed = _assistant("This is not JSON")
    repaired = _assistant('{"render":true,"title":"Price summary","components":[{"id":"m1","type":"metric","label":"Average","value":"48 €/MWh"}]}')
    fake = FakeLLM([malformed, repaired])
    surface = build_dynamic_ui(fake, "Compare prices", "Average is 48 €/MWh.", [])
    assert fake.calls == 2
    assert len(surface) == 2
    assert surface[1]["updateComponents"]["components"][-1]["id"] == "chat_genui_root"

def test_example_plan_is_valid_and_uses_only_catalog_components():
    from pathlib import Path
    example = json.loads((Path(__file__).resolve().parent.parent / "examples" / "sample_dynamic_ui_plan.json").read_text(encoding="utf-8"))
    messages = ui_tree_to_a2ui(example)
    components = messages[1]["updateComponents"]["components"]
    assert {c["component"] for c in components} >= {"Metric", "Chart", "Calculator", "AskAgentButton"}


# --- Conversation memory: the last 10 question/answer exchanges ---------------

def _exchanges(count, size=20):
    history = []
    for i in range(count):
        history.append({"role": "user", "content": f"q{i} " + "x" * size})
        history.append({"role": "assistant", "content": f"a{i} " + "y" * size})
    return history


def test_last_ten_exchanges_are_sent_to_the_model_in_order():
    from energy_agent.agent import MAX_HISTORY_TURNS
    assert MAX_HISTORY_TURNS == 10
    fake = FakeLLM([_assistant("Answer."), _assistant('{"render":false}')])
    Agent(fake, FakeTools()).run("current question", conversation_context=_exchanges(15))
    sent = fake.messages_seen[0]
    assert [m["role"] for m in sent[:1]] == ["system"]
    history = sent[1:-1]
    assert len(history) == 20                       # 10 questions + 10 answers
    assert history[0]["content"].startswith("q5 ")  # exchanges 0-4 fall out of the window
    assert history[-1]["content"].startswith("a14 ")
    assert sent[-1] == {"role": "user", "content": "current question"}


def test_history_budget_drops_oldest_messages_first():
    from energy_agent import agent as agent_module
    huge = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "z" * 5000}
            for i in range(20)]
    kept = Agent._history_messages(huge)
    assert sum(len(m["content"]) for m in kept) <= agent_module.MAX_HISTORY_TOTAL_CHARS
    assert kept[-1]["content"].startswith("m19 ")   # newest always survives
    assert len(kept) < 20                           # oldest were dropped


def test_history_ignores_malformed_items():
    kept = Agent._history_messages([
        {"role": "system", "content": "ignored"}, {"role": "user", "content": "  "},
        {"role": "user", "content": 5}, "not a dict", {"role": "user", "content": "kept"},
    ])
    assert kept == [{"role": "user", "content": "kept"}]


# --- Overfitting guard ---------------------------------------------------------
# The agent must stay general. Prompts, tool schemas and UI-planner instructions
# must not embed a specific user question, calendar date, appliance or clock window.

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OVERFIT_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}",                       # fixed ISO dates
    r"\b\d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December) 20\d\d",
    r"washing machine|dishwasher|dryer|tumble|heat pump|\bEV\b|electric vehicle",
    r"\b\d{1,2}:\d{2}\s*[–-]\s*\d{1,2}:\d{2}",  # fixed clock windows such as 18:00–21:00
]


def _runtime_prompt_text():
    from energy_agent import agent as agent_module, genui, tools
    import inspect
    planner_source = inspect.getsource(genui.build_dynamic_ui)
    return "\n".join([agent_module.SYSTEM_PROMPT, json.dumps(tools.TOOL_SCHEMAS), planner_source])


@pytest.mark.parametrize("pattern", _OVERFIT_PATTERNS)
def test_runtime_prompts_and_tool_schemas_contain_no_question_specific_content(pattern):
    assert re.search(pattern, _runtime_prompt_text(), flags=re.I) is None


@pytest.mark.parametrize("pattern", _OVERFIT_PATTERNS)
def test_frontend_source_contains_no_question_specific_content(pattern):
    for path in (_ROOT.parent / "webapp" / "src").glob("*.ts*"):
        assert re.search(pattern, path.read_text(encoding="utf-8"), flags=re.I) is None, path.name
