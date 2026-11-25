"""Dynamic, chat-native GenUI planning and A2UI v0.9.1 message assembly.

The model emits a small JSON UI tree, never executable code. This module validates
that tree against the server-side catalog and converts it into A2UI-style surface
messages. The React chat renderer is the trusted implementation of this catalog.
"""
from __future__ import annotations


import json
import math
import os
import re
from typing import Any



CATALOG_ID = "urn:energy-agent:chat-genui:catalog:v1"
SURFACE_ID = "chat-ui"
MAX_COMPONENTS = 36
MAX_DEPTH = 5
MAX_TABLE_ROWS = 80
MAX_CHART_POINTS = 100

TEXT_VARIANTS = {"body", "caption", "title", "heading", "code"}
CHART_TYPES = {"line", "bar", "area", "scatter"}
CALC_OPERATIONS = {"energy_cost", "multiply", "add", "subtract", "divide", "percentage"}



class UIValidationError(ValueError):
    pass



def _text(value: Any, limit: int = 5000) -> str:
    if value is None:
        return ""
    if not isinstance(value, (str, int, float, bool)):
        raise UIValidationError("Text fields must be scalar values.")
    value = str(value)
    if len(value) > limit:
        value = value[:limit]
    return value



def _number(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise UIValidationError("Expected a finite numeric value.")
    if minimum is not None and value < minimum:
        raise UIValidationError(f"Number must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise UIValidationError(f"Number must be <= {maximum}.")
    return float(value)


def validate_ui_tree(tree: Any) -> dict[str, Any]:
    """Validate the LLM-generated UI tree, clamp its size, and return normalized JSON."""
    if not isinstance(tree, dict) or not isinstance(tree.get("components"), list):
        raise UIValidationError("UI must be an object containing a components array.")
    if len(tree["components"]) > MAX_COMPONENTS:
        raise UIValidationError(f"UI may contain at most {MAX_COMPONENTS} top-level components.")
    ids: set[str] = {"chat_genui_root", "genui_title"}
    total = 0

    def walk(node: Any, depth: int = 1) -> dict[str, Any]:
        nonlocal total
        if depth > MAX_DEPTH:
            raise UIValidationError("UI nesting exceeds maximum depth.")
        if not isinstance(node, dict):
            raise UIValidationError("Each UI component must be an object.")
        total += 1
        if total > MAX_COMPONENTS:
            raise UIValidationError(f"UI may contain at most {MAX_COMPONENTS} components in total.")
        cid = _text(node.get("id") or f"node-{total}", 64)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cid):
            raise UIValidationError("Component IDs must use letters, digits, underscore, or hyphen.")
        if cid in ids:
            raise UIValidationError(f"Duplicate component id: {cid}")
        ids.add(cid)
        kind = node.get("type")
        if kind not in {"text", "card", "row", "column", "list", "metric", "chart", "table", "slider", "calculator", "button", "callout"}:
            raise UIValidationError(f"Unsupported component type: {kind}")
        out: dict[str, Any] = {"id": cid, "type": kind}

        if kind == "text":
            out["text"] = _text(node.get("text"), 5000)
            variant = node.get("variant", "body")
            if variant not in TEXT_VARIANTS:
                raise UIValidationError("Unsupported text variant.")
            out["variant"] = variant
        elif kind in {"card", "row", "column"}:
            if kind == "card":
                synthetic = f"{cid}_content"
                if synthetic in ids:
                    raise UIValidationError(f"Reserved synthetic component id collision: {synthetic}")
                ids.add(synthetic)
            out["title"] = _text(node.get("title", ""), 200)
            children = node.get("children", [])
            if not isinstance(children, list) or len(children) > 16:
                raise UIValidationError("Container children must be an array of at most 16 items.")
            out["children"] = [walk(child, depth + 1) for child in children]
        elif kind == "list":
            items = node.get("items", [])
            if not isinstance(items, list) or len(items) > 40:
                raise UIValidationError("List must contain at most 40 items.")
            out["items"] = [_text(item, 1000) for item in items]
        elif kind == "metric":
            out.update(label=_text(node.get("label"), 150), value=_text(node.get("value"), 150),
                       detail=_text(node.get("detail", ""), 500))
        elif kind == "chart":
            chart_type = node.get("chart_type", "line")
            if chart_type not in CHART_TYPES:
                raise UIValidationError("Unsupported chart type.")
            x_key = _text(node.get("x_key", "x"), 64)
            data = node.get("data", [])
            series = node.get("series", [])
            if not isinstance(data, list) or len(data) > MAX_CHART_POINTS:
                raise UIValidationError(f"Chart data must contain at most {MAX_CHART_POINTS} points.")
            if not isinstance(series, list) or not 1 <= len(series) <= 6:
                raise UIValidationError("Chart must have 1–6 series.")
            normalized_series = []
            keys = {x_key}
            for s in series:
                if not isinstance(s, dict):
                    raise UIValidationError("Chart series must be objects.")
                key = _text(s.get("key"), 64)
                if not re.fullmatch(r"[A-Za-z0-9_ -]{1,64}", key):
                    raise UIValidationError("Invalid chart series key.")
                if key in keys:
                    raise UIValidationError("Chart series keys must be unique and differ from x_key.")
                keys.add(key)
                normalized_series.append({"key": key, "label": _text(s.get("label", key), 100)})
            normalized_data = []
            for point in data:
                if not isinstance(point, dict):
                    raise UIValidationError("Chart data points must be objects.")
                x_value = point.get(x_key)
                row = {x_key: _number(x_value) if chart_type == "scatter" else _text(x_value, 100)}
                for s in normalized_series:
                    value = point.get(s["key"])
                    if value is not None:
                        row[s["key"]] = _number(value)
                normalized_data.append(row)
            out.update(title=_text(node.get("title", ""), 200), chart_type=chart_type,
                       x_key=x_key, series=normalized_series, data=normalized_data)
        elif kind == "table":
            columns, rows = node.get("columns", []), node.get("rows", [])
            if not isinstance(columns, list) or not 1 <= len(columns) <= 8:
                raise UIValidationError("Table must have 1–8 columns.")
            if not isinstance(rows, list) or len(rows) > MAX_TABLE_ROWS:
                raise UIValidationError(f"Table must have at most {MAX_TABLE_ROWS} rows.")
            normalized_cols = []
            col_keys = set()
            for col in columns:
                if not isinstance(col, dict):
                    raise UIValidationError("Table columns must be objects.")
                key = _text(col.get("key"), 64)
                if key in col_keys:
                    raise UIValidationError("Table column keys must be unique.")
                col_keys.add(key)
                normalized_cols.append({"key": key, "label": _text(col.get("label", key), 100)})
            normalized_rows = []
            for row in rows:
                if not isinstance(row, dict):
                    raise UIValidationError("Table rows must be objects.")
                normalized_rows.append({k: _text(row.get(k), 1000) for k in col_keys})
            out.update(title=_text(node.get("title", ""), 200), columns=normalized_cols, rows=normalized_rows)
        elif kind == "slider":
            minimum = _number(node.get("min", 0))
            maximum = _number(node.get("max", 100))
            step = _number(node.get("step", 1), minimum=0.0000001)
            value = _number(node.get("value", minimum), minimum=minimum, maximum=maximum)
            if maximum <= minimum:
                raise UIValidationError("Slider max must be greater than min.")
            out.update(label=_text(node.get("label"), 150), min=minimum, max=maximum, step=step,
                      value=value, unit=_text(node.get("unit", ""), 24))
        elif kind == "calculator":
            operation = node.get("operation")
            if operation not in CALC_OPERATIONS:
                raise UIValidationError("Unsupported calculator operation.")
            fields = node.get("fields", [])
            if not isinstance(fields, list) or not 2 <= len(fields) <= 4:
                raise UIValidationError("Calculator must contain 2–4 input fields.")
            normalized_fields = []
            names = set()
            for field in fields:
                if not isinstance(field, dict):
                    raise UIValidationError("Calculator fields must be objects.")
                name = _text(field.get("name"), 32)
                if not re.fullmatch(r"[A-Za-z0-9_]{1,32}", name) or name in names:
                    raise UIValidationError("Calculator field names must be unique simple identifiers.")
                names.add(name)
                low = _number(field.get("min", 0))
                high = _number(field.get("max", 1000000))
                step = _number(field.get("step", 1), minimum=0.0000001)
                value = _number(field.get("value", low), minimum=low, maximum=high)
                if high <= low:
                    raise UIValidationError("Calculator field max must exceed min.")
                normalized_fields.append({"name": name, "label": _text(field.get("label", name), 100),
                                        "min": low, "max": high, "step": step, "value": value,
                                        "unit": _text(field.get("unit", ""), 24)})
            if operation == "energy_cost" and not {"power_kw", "duration_hours", "price_eur_mwh"}.issubset(names):
                raise UIValidationError("energy_cost requires power_kw, duration_hours and price_eur_mwh fields.")
            if operation == "energy_cost":
                for field in normalized_fields:
                    if field["name"] in {"power_kw", "duration_hours"} and field["min"] < 0:
                        raise UIValidationError(f"{field['name']} must have a nonnegative minimum.")
            out.update(title=_text(node.get("title", "Calculator"), 200), operation=operation,
                      fields=normalized_fields, output_label=_text(node.get("output_label", "Result"), 100),
                      output_unit=_text(node.get("output_unit", ""), 32))
        elif kind == "button":
            if node.get("action") != "ask_agent":
                raise UIValidationError("Only ask_agent buttons are supported.")
            prompt = _text(node.get("prompt"), 1000)
            if not prompt.strip():
                raise UIValidationError("Button prompt cannot be empty.")
            out.update(label=_text(node.get("label", "Ask"), 100), action="ask_agent", prompt=prompt)
        elif kind == "callout":
            tone = node.get("tone", "info")
            if tone not in {"info", "success", "warning", "error"}:
                raise UIValidationError("Invalid callout tone.")
            out.update(title=_text(node.get("title", ""), 200), text=_text(node.get("text"), 3000), tone=tone)
        return out

    normalized = {"title": _text(tree.get("title", ""), 200),
                 "components": [walk(node) for node in tree["components"]]}
    return normalized



def _convert_component(node: dict[str, Any], flat: list[dict[str, Any]]) -> str:
    """Flatten validated nested component tree into A2UI component IDs + refs."""
    cid, kind = node["id"], node["type"]
    if kind in {"card", "row", "column"}:
        children = [_convert_component(child, flat) for child in node["children"]]
        if kind == "card":
            container_id = f"{cid}_content"
            flat.append({"id": container_id, "component": "Column", "children": children})
            props = {"id": cid, "component": "Card", "child": container_id}
            if node.get("title"):
                props["title"] = node["title"]
            flat.append(props)
        else:
            flat.append({"id": cid, "component": "Row" if kind == "row" else "Column", "children": children})
    elif kind == "text":
        flat.append({"id": cid, "component": "Text", "text": node["text"], "variant": node["variant"]})
    else:
        # Custom catalog components are implemented in the chat React renderer.
        props = {k: v for k, v in node.items() if k != "type"}
        props["component"] = {"metric": "Metric", "chart": "Chart", "table": "DataTable",
                              "list": "List", "slider": "Slider", "calculator": "Calculator",
                              "button": "AskAgentButton", "callout": "Callout"}[kind]
        flat.append(props)
    return cid



def ui_tree_to_a2ui(tree: dict[str, Any]) -> list[dict[str, Any]]:
    """Create v0.9.1-compatible envelopes for the app's versioned custom catalog.

    A custom React renderer is used because the maintained official React renderer
    currently advertises v0.9.1 stable support, while v1.0 is not yet marked stable.
    """
    validated = validate_ui_tree(tree)
    flat: list[dict[str, Any]] = []
    root_children = []
    if validated.get("title"):
        flat.append({"id": "genui_title", "component": "Text", "text": validated["title"], "variant": "heading"})
        root_children.append("genui_title")
    root_children.extend(_convert_component(c, flat) for c in validated["components"])
    flat.append({"id": "chat_genui_root", "component": "Column", "children": root_children})
    messages = [
        {"version": "v0.9", "createSurface": {"surfaceId": SURFACE_ID, "catalogId": CATALOG_ID}},
        {"version": "v0.9", "updateComponents": {"surfaceId": SURFACE_ID, "components": flat}},
    ]
    return messages



def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def build_dynamic_ui(llm: Any, query: str, answer: str, observations: list[dict[str, Any]]) -> list[dict]:
    """Generate an optional evidence-aware UI with one bounded repair attempt."""
    obs = json.dumps(observations, ensure_ascii=False, default=str)[:14000]
    system = """You are the dynamic inline-chat UI planner for an energy-price assistant. You do not create webpages, HTML, CSS, JavaScript, or code. Decide if a visual UI materially improves the response. If not, return exactly {\"render\":false}. If yes, return JSON only: {\"render\":true,\"title\":string,\"components\":array}. Compose a useful UI from the available component catalog, dynamically based on the specific user question and actual answer/evidence. Avoid template layouts and avoid repeating the prose answer. Never invent data: use only values present in answer or tool observations. Components allowed: text {id,type,text,variant body|caption|title|heading|code}; card {id,type,title?,children}; row/column {id,type,title?,children}; metric {id,type,label,value,detail?}; chart {id,type,title,chart_type line|bar|area|scatter,x_key,series:[{key,label}],data:[{x-key:string,series numeric values}]}; table {id,type,title,columns:[{key,label}],rows:[objects]}; list {id,type,items:[strings]}; slider {id,type,label,min,max,step,value,unit?}; calculator {id,type,title,operation energy_cost|multiply|add|subtract|divide|percentage,fields:[{name,label,min,max,step,value,unit?}],output_label,output_unit}; button {id,type,label,action:'ask_agent',prompt}; callout {id,type,title,text,tone info|success|warning|error}. Components can be nested in card/row/column. Use 1–8 components when a UI helps; the hard maximum is 36, with max depth 5, max 100 chart points and max 80 table rows. For an interactive calculator, use only supported safe operations, never executable expressions. Sliders are independent and must not imply recalculation unless a calculator component is used. Output valid JSON, no Markdown fences."""
    base_user = f"USER QUESTION:\n{query}\n\nFINAL ANSWER:\n{answer}\n\nTOOL OBSERVATIONS:\n{obs}\n\nGenerate a suitable UI or no UI."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": base_user}]
    last_error = ""
    last_output = ""
    # Plan JSON deterministically. Some OpenAI-compatible servers reject
    # response_format, so allow it to be disabled with UI_PLANNER_JSON_MODE=0.
    json_mode = os.getenv("UI_PLANNER_JSON_MODE", "1") != "0"
    for attempt in range(2):
        try:
            response = llm.generate(messages=messages, tools=[],
                                    temperature=0.0, json_mode=json_mode)
            if response.tool_calls:
                raise UIValidationError("UI planning response must not contain tool calls.")
            last_output = response.content or ""
            parsed = _extract_json(last_output)
            if not isinstance(parsed, dict) or parsed.get("render") is not True:
                return []
            parsed.setdefault("title", "")
            if not isinstance(parsed.get("components"), list) or not parsed["components"]:
                return []
            return ui_tree_to_a2ui(parsed)
        except Exception as exc:
            last_error = str(exc)
            if attempt == 0:
                messages.append({"role": "assistant", "content": last_output})
                messages.append({"role": "user", "content": (
                    "Repair the UI plan only. Your previous output failed validation.\n"
                    f"Validation error: {last_error}\nPrevious output:\n{last_output[:8000]}\n"
                    "Return a corrected JSON plan or {\"render\":false}. Do not add new factual data.")})
    return []
