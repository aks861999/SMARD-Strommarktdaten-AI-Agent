# CLAUDE.md — Energy Price Planning Agent

Tool-using agent for the German DE-LU day-ahead electricity market. Three services, each in its own
folder under `services/`: `api` (FastAPI + SSE agent), `prices` (hourly prices behind a cache) and `webapp`
(React chat UI + nginx). See `README.md` for setup.

## Commands

```powershell
pip install -r requirements-dev.txt                   # all services' runtime deps + test deps
python -m pytest -q                                   # every service's tests (test_api.py needs httpx2)
python -m uvicorn api:app --app-dir services/api --host 127.0.0.1 --port 8000
cd services/webapp; npm run dev                       # http://127.0.0.1:5173
cd services/webapp; npx tsc --noEmit                  # typecheck
docker compose up --build                             # the whole stack, http://localhost:8080 (run in WSL on Windows)
```

## Rule #1 — Keep the system general (no overfitting)

The agent must answer **any** question inside its domain (prices on a day, price pattern, cheapest
window, wholesale run cost), not one demo question. Never tune the system to a single prompt.

- **Do not** put a specific user question, calendar date, appliance, power rating or clock window
  (e.g. "2 kW washing machine, 16 November, 18:00–21:00") into the system prompt, tool schemas,
  UI-planner prompt, starter suggestions, defaults or frontend code.
- **Do not** add keyword/regex routing, per-question `if` branches, or canned answers. Fix
  behaviour by improving the general prompt rule, the tool description, or the tool's validation.
- **Do not** make a test pass by special-casing its input. Tests may use fixed dates/values as
  *fixtures* in the services' `tests/` folders, but runtime code must never reference them.
- The model resolves relative dates ("tomorrow") from the current Berlin date injected in
  `Agent._system_message`; never hard-code "today".
- Starter suggestions in `ChatApp.tsx` only prefill the input. Keep them few, varied in intent, and
  free of fixed dates. One is deliberately a multi-step request ("Detailed price analysis") so the
  agent's multi-tool reasoning stays easy to demo; keep it relative ("tomorrow") and appliance-neutral.
- `services/api/tests/test_agent_architecture.py` contains an overfitting guard that scans prompts, tool
  schemas and `services/webapp/src` for fixed dates, appliances and clock windows. If it fails, remove the
  hard-coded content — do not loosen the regexes.
- Verify changes against **varied** phrasings and intents (different dates, durations, appliances,
  follow-ups, missing inputs, out-of-domain questions), not just the one you are debugging.

## Agentic best practices followed here

Sources: Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
and [Writing tools for agents](https://anthropic.com/engineering/writing-tools-for-agents);
OpenAI, [A practical guide to building agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/).

1. **Simplest thing that works.** Plain LLM + tool loop, no agent framework. Add complexity only
   when an evaluation shows it is needed.
2. **Tools are an interface to design (ACI).** Clear names, precise parameter docs and bounds,
   explicit units (EUR/MWh), mistake-proof arguments (`enum`, `minimum`/`maximum`), actionable error
   results. Unsupported or unknown arguments are rejected in `Agent._execute_tool`.
3. **Ground truth every step.** Numbers come from tool results (SMARD or labelled demo data), never
   from the model. Missing data is reported, not estimated. State the wholesale-only limitation.
4. **Stopping conditions and guardrails.** Runtime limit (`AGENT_MAX_RUNTIME_SECONDS`), optional
   step ceiling, duplicate-tool-call caching, bounded concurrency, cancellation.
5. **Transparency.** The collapsed "Behind the scenes" trace shows decisions, tool arguments and
   observations. Provider `reasoning_content` is never sent to the client.
6. **Untrusted model output is validated.** The UI planner emits JSON only; `genui.py` enforces the
   catalog, size and depth limits. Never render model-generated HTML/JS. UI is optional and its
   failure must never discard a finished text answer.
7. **Bounded, explicit memory.** The last **10 question/answer exchanges (20 messages)** are always
   sent (`MAX_HISTORY_TURNS` in `agent.py`, mirrored by `HISTORY_TURNS` in `ChatApp.tsx`), with
   per-message and total character budgets that drop the oldest first. There is no opt-out toggle.
   Resolved slots are carried separately as small, typed, validated state.
8. **Progressive delivery.** Stream the finished text first, append optional UI afterwards. Do not
   make a complete answer wait on an optional second model call.
9. **Evaluate, don't eyeball.** Prefer adding a general regression test or eval case over a one-off
   prompt tweak. Keep a held-out set of varied questions separate from the ones used to tune prompts.

## Architecture rules

- **LLM provider is backend-only.** Chosen by `LLM_PROVIDER` (default `nvidia`). The dashboard has
  no provider selector and the API ignores a client-supplied `provider`. Do not re-add one.
- **One history limit.** `MAX_HISTORY_MESSAGES` in `services/api/energy_agent/agent.py` is imported by `api.py`;
  change it in one place and keep `HISTORY_TURNS` in the frontend in sync.
- Do not stream tool-iteration text to the chat; only the final no-tool answer is shown.
- **Keep the UI mobile friendly.** Below 760px the sidebar is hidden and "New chat" lives in the top bar;
  text fields are 16px (iOS zooms into anything smaller); touch targets are about 40px or more; wide
  tables scroll sideways inside their box instead of widening the page; safe-area insets pad notches.
  On touch devices Enter adds a new line and the button sends. Check a change at 390px and 320px wide.
- **Services and Docker.** Each service is self-contained in `services/<name>/` with its own `Dockerfile` and
  `requirements.txt`, and `docker-compose.yml` follows the plain style of the reference project: a comment
  per service, `build: context: ./services/<name>`, explicit `${VAR:-default}` environment lists, one named
  network. `webapp` (nginx: static React + `/api` proxy) is the only service that publishes a port; `api`
  (agent) gets the NVIDIA key and only the variables compose lists; `prices` (SMARD/demo data behind
  `prices_service/cache.py`) has no secrets. The services share no code at run time: `api` talks to `prices`
  over HTTP (`HttpPriceSource`). Single-service mode (Render, local dev, `PRICES_URL` unset): `api` imports
  the price code from `services/prices` of the same checkout, so never deploy `services/api` alone without
  `PRICES_URL`. `.env` is never copied into an image (`.dockerignore` in each service). nginx must SET
  `X-Forwarded-For`, never append to a client's. An empty `VITE_API_BASE_URL` means same origin, so the
  frontend uses `??`, not `||`. Never cache failures. Text files keep LF endings (`.gitattributes`).
  The local llama.cpp model and Redis are deliberately not part of the stack.
- **Rate limiting** (`energy_agent/ratelimit.py`, `_admit` in `api.py`): new questions are limited per
  client IP, with a global backstop; resuming an existing `request_id` is never counted and a rejected
  request spends no budget. Keep the limiter free of FastAPI. Never trust the leftmost `X-Forwarded-For`
  entry or any header the operator has not named in `CLIENT_IP_HEADER`. The frontend must show the
  server's 429 message and must not retry it in a loop.
- Adding a UI component means updating `services/api/catalog.json`, `genui.py` validation, `ui_tree_to_a2ui`,
  `services/webapp/src/renderer.tsx`, and tests together.
- **Deployment:** backend on Render free tier (`render.yaml`), frontend on Vercel; see the README
  "Deployment" section. Keep a single uvicorn process (job state is in memory). Keep
  each service's `requirements.txt` runtime-only; test packages go in the root `requirements-dev.txt`.
  Vercel's Root Directory is `services/webapp`. The frontend must
  retry the initial POST (not the resume endpoint) until the server has accepted the job.
- Never commit or print `.env` (it holds `NVIDIA_API_KEY`). Use `example.env` for new settings.
- Local-development codebase: no auth or durable storage. Keep the API bound to localhost.

## Known overfitting risks to watch

- `Agent._merge_state` carries slots (`date`, `power_kw`, `duration_hours`, …) between turns. This
  is general, but a stale slot can leak into an unrelated question. The prompt tells the model to
  reuse state only for follow-ups; if leakage shows up, tighten that rule rather than special-casing.
- `presentation/` (git-ignored) and `services/api/examples/sample_dynamic_ui_plan.json` are illustrations of one
  scenario. They are not runtime inputs; do not copy their wording into prompts.
