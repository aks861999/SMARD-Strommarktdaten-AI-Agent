# Energy Price Planning Agent, Chat-native Dynamic GenUI


A tool-using agent for the German DE-LU day-ahead electricity market. This final codebase replaces the Streamlit-only interface with a React chat frontend. The assistant text streams into the chat; when a UI adds value, a separate UI-planner call generates an optional, validated, **inline** component surface. It does not generate or open webpages.


## What is implemented

- **Dynamic UI planning:** no fixed energy dashboard. The UI planner receives the user question, final answer, and tool observations, then either returns `render:false` or chooses a UI composition.
- **Chat-native UI:** A2UI-style `createSurface` and `updateComponents` messages are rendered inside the assistant transcript by a project-owned React catalog renderer.
- **Component catalog:** Text, Row, Column, Card, Metric, Chart (line/bar/area/scatter), DataTable (click-to-sort), List, local Slider, safe Calculator (whitelisted operations), AskAgentButton (follow-up only), and Callout.
- **Streaming and reconnect:** final answer text, agent trace and UI events are sent over SSE with monotonically increasing event IDs. The API retains a bounded in-memory event log for one hour so the browser can replay missed events after a transient connection interruption. A2UI surface messages remain separate rendering events.
- **Local interactions:** sliders and calculators update immediately in the browser. Ask-agent buttons send an ordinary follow-up prompt to the backend. The text transcript supports GitHub-Flavoured Markdown, including rendered tables.
- **Bounded validation:** supported types/operations only; unique IDs; limits on nesting, component count, chart data and table rows; no generated JS/HTML execution. One repair attempt is made for invalid UI output; if it still fails, the text answer remains.
- **Tool deduplication:** repeated identical tool calls are cached within a single agent turn.
- **Stop/cancel:** the Stop button calls an explicit API cancellation endpoint. A transient SSE disconnect does not cancel the agent; the browser reconnects and replays missed events. Cancellation is checked between provider/tool steps; an already-running provider HTTP request cannot be forcibly terminated and may continue until its request timeout.
- **Conversation context:** previous messages remain in browser memory for the current page session and the last 10 question/answer exchanges (20 messages) are **always** sent with each request; there is no switch to turn this off (a character budget drops the oldest messages first if the window is very long). Resolved slots from the previous turn (`date`, `duration_hours`, `power_kw`, …) are returned in the `done` event and resent as `state`, so follow-ups such as “make it four hours instead” keep their referents. This is not durable or server-side memory; use **New chat** to clear both messages and carried state.
- **Agent trace:** the collapsed “Behind the scenes · Agent execution” section displays iteration decisions, tool names, tool arguments and structured tool observations emitted by `agent.py`. It deliberately excludes provider `reasoning_content` and A2UI protocol payloads.
- **Providers:** hosted NVIDIA NIM (default) or a local llama.cpp server through the OpenAI-compatible endpoint. The provider is **backend configuration only** (`LLM_PROVIDER`); the dashboard shows no provider selector.
- **Data:** live SMARD is default; synthetic demo CSV can be selected.


## Architecture

```mermaid
flowchart TD
  U[User prompt] --> F[React chat frontend]
  F -->|POST /api/chat| API[FastAPI SSE API]
  API --> A[Energy Agent]
  A <--> L[OpenAI-compatible LLM adapter]
  A --> T[Validated energy tools]
  T --> S[SMARD or synthetic source]
  A -->|final answer| TEXT[Text stream]
  A -->|answer + observations| P[Dynamic UI Planner]
  P --> V[Tree validation + one repair retry]
  V -->|A2UI v0.9 envelopes| SSE[SSE events]
  TEXT --> CHAT[Chat message renderer]
  SSE --> CHAT
  CHAT --> C[Versioned React catalog]
  C --> INLINE[Inline cards, charts, tables, controls]
  INLINE -->|AskAgentButton follow-up| API
```

Delivery is always **progressive**: the text answer is shown as soon as it is ready and the optional UI is appended when planning completes. A "wait for text + UI" mode was evaluated and removed: the UI needs a second, optional model call, so holding a finished answer back for it only adds latency and makes a UI failure delay (or hide) the answer. A turn has two stages: (1) the tool/decision loop, which is never streamed so planning text cannot leak into the transcript, and (2) an optional UI-planner call. The model's first response without a tool call is the answer itself; it is generated once and emitted to the chat, so there is no second full answer call and no risk of two generations disagreeing. The agent carries resolved slots from the previous turn into the system prompt so follow-ups resolve against the right date and assumptions. UI generation adds a bounded repair call only when the first UI plan fails parsing or validation. The UI planner uses the evidence in the turn; the prompt asks it not to invent data, but semantic factual grounding of all UI prose is not mechanically proven.


## A2UI implementation note

The project emits the `v0.9` wire envelopes (`createSurface`, `updateComponents`) with a versioned custom catalog ID. The official maintained A2UI React renderer is **not installed** here. Instead, the project includes a custom React renderer for the component catalog, including its chart/table/calculator extensions. This means:

- It is a project-owned chat-native renderer for this A2UI-style subset and custom catalog.
- It does not execute generated HTML/JavaScript.
- It does not claim full official A2UI schema conformance or compatibility with an unmodified renderer.
- `catalog.json` is the catalog contract. Add components there, then implement their renderer and validation together.

The official A2UI renderer documentation currently describes the shared web core and renderer implementation model; check the compatibility matrix before switching protocol/renderer versions. Official React renderer support and A2UI v1.0 availability have not been browser-verified in this codebase.


## Project layout

Every service lives in its own folder under `services/`, with its own Dockerfile and requirements.

```text
.
├── docker-compose.yml          # webapp + api + prices, one command
├── example.env                 # copy to .env; docker compose and the api read it
├── render.yaml                 # Render blueprint (the single-service deployment)
├── pytest.ini                  # one `python -m pytest` runs every service's tests
├── requirements-dev.txt        # test dependencies for all services
└── services/
    ├── api/                    # the agent
    │   ├── api.py              # FastAPI endpoint, SSE, rate limiting, cancellation
    │   ├── main.py             # CLI entry point / LLM provider setup
    │   ├── catalog.json        # Versioned chat component catalog manifest
    │   ├── energy_agent/
    │   │   ├── agent.py        # Tool loop, dedupe, event emission
    │   │   ├── genui.py        # Dynamic UI planner, validation, A2UI message assembly
    │   │   ├── llm.py          # Streaming OpenAI-compatible adapters
    │   │   ├── tools.py        # Energy tool schemas and deterministic functions
    │   │   ├── ratelimit.py    # Per-IP rate limiting
    │   │   └── sources.py      # PriceSource interface, HTTP client of the prices service
    │   ├── examples/           # Example JSON UI plan (synthetic only)
    │   ├── tests/
    │   ├── Dockerfile
    │   └── requirements.txt
    ├── prices/                 # hourly prices behind a cache
    │   ├── prices_service/
    │   │   ├── app.py          # GET /prices, GET /health
    │   │   ├── sources.py      # SMARD and demo sources
    │   │   └── cache.py        # thread-safe TTL cache
    │   ├── data/sample_prices.csv
    │   ├── tests/
    │   ├── Dockerfile
    │   └── requirements.txt
    └── webapp/                 # the React chat UI
        ├── src/                # Chat UI + custom A2UI catalog renderer
        ├── docker/nginx.conf.template
        ├── package.json
        ├── vite.config.ts
        └── Dockerfile
```

The services share no code at run time. The api asks the prices service for data over HTTP. The one exception is
the single-service mode (Render, local development): with `PRICES_URL` unset, the api imports the price code from
`services/prices` in the same checkout, so it needs a full copy of the repository there.


## Quickstart (Windows PowerShell)

### 1. Python backend

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
# each service has its own requirements.txt (runtime only); requirements-dev.txt installs all of them
# plus the test dependencies (pytest, httpx2).
Copy-Item example.env .env
```

Edit `.env` and add `NVIDIA_API_KEY` (NVIDIA NIM is the default provider). To use a local llama.cpp server instead, start it and set `LLM_PROVIDER=llama-cpp` in `.env`; the dashboard has no provider selector.

Start the backend:

```powershell
python -m uvicorn api:app --app-dir services/api --host 127.0.0.1 --port 8000
```

### 2. React chat frontend (second terminal)

```powershell
cd services/webapp
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

### 3. Demo data

In `.env`, set:

```dotenv
PRICE_SOURCE=demo
```

Restart the backend. Synthetic demo values are labelled in tool results; don't treat them as live market prices.

### 4. CLI remains available

```powershell
python services/api/main.py --provider nvidia --data smard
# or
python services/api/main.py --provider llama-cpp --data demo
```

### 5. Python tests

```powershell
python -m pytest -q        # every service, from the repository root
```


## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `nvidia` | Backend-only provider choice: `nvidia` or `llama-cpp` |
| `NVIDIA_API_KEY` | — | Required for hosted NIM |
| `NVIDIA_BASE_URL` | NVIDIA endpoint | OpenAI-compatible API URL |
| `NEMOTRON_MODEL` | project default | Hosted model ID |
| `NEMOTRON_REASONING_EFFORT` | `low` | Reasoning setting forwarded to provider |
| `NEMOTRON_MAX_TOKENS` | `8192` | Hosted output budget |
| `PRICE_SOURCE` | `smard` | `smard` or `demo` |
| `UI_PLANNER_JSON_MODE` | `1` | Send `response_format=json_object` to the UI planner; set `0` if your endpoint rejects it |
| `AGENT_MAX_STEPS` | `0` | Optional iteration-count ceiling; `0` means no count limit |
| `AGENT_MAX_RUNTIME_SECONDS` | `900` | Runtime safety limit; `0` disables it |
| `MAX_CONCURRENT_AGENT_RUNS` | `2` | Backend concurrency slots |
| `RATE_LIMIT_PER_MINUTE` | `6` | New questions allowed per client IP per minute; `0` = off |
| `RATE_LIMIT_PER_HOUR` | `60` | New questions allowed per client IP per hour; `0` = off |
| `RATE_LIMIT_GLOBAL_PER_HOUR` | `300` | New questions per hour across all clients (a backstop that faking an IP cannot dodge); `0` = off |
| `RATE_LIMIT_MAX_ACTIVE_PER_IP` | `1` | Runs one client may have going at once (a stopped run does not count); `0` = off |
| `CLIENT_IP_HEADER` | unset | Header holding the visitor IP behind a proxy, e.g. `cf-connecting-ip`; unset = use the connection address |
| `TRUSTED_PROXY_HOPS` | `1` | Only with `CLIENT_IP_HEADER=x-forwarded-for`: how many trusted proxies sit in front of the app |
| `CHAT_CORS_ORIGINS` | local Vite origins | Comma-separated browser origins |
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Frontend API endpoint, read at build time; set in `services/webapp/.env.local` if needed. Empty means same origin (the Docker setup) |
| `PRICES_URL` | unset | Address of the prices service. Set = the API gets prices from it (Docker); unset = the API fetches them itself |
| `WEBAPP_PORT` | `8080` | Docker only: host port for the webapp service |
| `WEBAPP_API_URL` | `http://api:8000` | Docker only: the API that nginx forwards `/api` to |


### Data and DST

German market days are not always 24 hours: the Berlin spring-forward day has 23 and the autumn fall-back day has 25. Tool completeness checks derive the expected hour count from the real local day, and window/cost calculations step through absolute hours, so repeated or skipped local hours are handled rather than reported as missing data. The demo source follows the same rule.


## Conversation context and debugging

Conversation messages are held in React component state only while the page remains open. The frontend always sends the last 10 question/answer exchanges (20 messages) as `history` plus the previous turn's resolved `state`. Use **New chat** to clear the visible transcript and the carried state. The **Behind the scenes** panel shows the agent execution trace emitted by `agent.py` (iteration decisions, tool arguments and tool observations); it does not expose A2UI protocol JSON or provider-internal reasoning. The agent has no iteration-count limit by default: it continues until the model stops requesting tools, subject to the 15-minute runtime safety limit. Set a positive `AGENT_MAX_STEPS` to enable an optional count ceiling. SSE events are buffered server-side briefly and can be replayed after a transient browser-stream interruption. A `request_id` reused with a different payload is rejected with HTTP 409 rather than attaching to the earlier answer.

## Current safety / operational limits

- This is a **local-development codebase**, not a public production deployment. It has no authentication, user accounts, durable conversation storage, distributed rate limiting, or reverse-proxy hardening. Keep the API bound to localhost until these are added.
- The model can propose UI components and values. Server validation enforces the catalog and bounds, but the UI planner's factual claims still require evaluation/grounding tests. Chart/table data are requested to be based on evidence.
- The generated UI cannot execute model-supplied code. Calculators use fixed operations, and interactive buttons can only submit a follow-up prompt.
- Python regression tests pass in the authoring environment. A frontend `npm install` / Vite build and browser end-to-end test could not be completed here because package installation timed out; frontend compilation is therefore **not verified** here. Dependencies are version-pinned in `package.json`; `npm install` on your machine will install `remark-gfm` and generate a lockfile.
- A2UI messages and custom catalog are validated by the project's own code/tests, not by the official A2UI schema validator. Before deploying beyond localhost, pin and integrate the official schemas/renderer or expand the integration tests.


## Run everything with Docker (three services)

```text
browser ──► webapp (nginx) ──► api (agent) ──► prices (SMARD + cache) ──► smard.de
              page + /api          │
              proxy                └──► NVIDIA API
```

| Service | Folder | What it does | Reachable from |
|---|---|---|---|
| `webapp` | `services/webapp` | nginx: serves the built React app and forwards `/api` to `api`, with streaming-safe settings | your machine, port 8080 (the only published port) |
| `api` | `services/api` | The agent loop, the four tools, the live stream, rate limiting and the UI planner. It alone gets `NVIDIA_API_KEY` | inside the stack only |
| `prices` | `services/prices` | Fetches SMARD (or demo) prices and caches them: 6 hours for a complete day, 2 minutes for an empty or partial one, failures never. Holds no secrets | inside the stack only |

```bash
cp example.env .env             # then put your NVIDIA_API_KEY in .env
docker compose up --build       # first build takes a few minutes
# open http://localhost:8080
```

- Use synthetic prices without the SMARD website: `PRICE_SOURCE=demo docker compose up --build`.
- Another port: `WEBAPP_PORT=9000 docker compose up --build`.
- Look around: `docker compose ps`, `docker compose logs -f api`, `curl http://localhost:8080/healthz`.
- Stop: `docker compose down`.
- The page and `/api` share one address, so no CORS setup is needed. nginx tells `api` the visitor's address (`X-Forwarded-For`), which the rate limiter uses.
- compose hands `api` only the variables it lists (not the whole `.env`), and the containers run as an unprivileged user. `.env` is never copied into an image.
- The optional local model (llama.cpp) is not part of this stack.
- This stack is for running on your own machine or a server you control. The Render and Vercel deployment below is separate: there the API fetches prices itself, because `PRICES_URL` is not set.
- On Windows with Docker inside WSL, run the commands from a WSL shell (the project is at `/mnt/d/...`).

## Deployment: backend on Render (free), frontend on Vercel

The backend is a long-lived FastAPI process (SSE streaming, in-memory jobs), so it needs a host that runs one, not Streamlit or serverless functions. `render.yaml` defines the free Render web service.

**1. Push the project folder to GitHub** (the repository root must be this folder; `.env` is git-ignored, never commit it).

**2. Backend on Render**
1. Render dashboard → **New → Blueprint** → select the repository. Render reads `render.yaml`: it builds the whole repository and starts the api from `services/api`, with the price code running inside it.
2. When asked, enter `NVIDIA_API_KEY` and `CHAT_CORS_ORIGINS`. For now put a placeholder origin; you set the real one in step 4.
3. Wait for the deploy and open `https://<your-service>.onrender.com/api/health`; it should return `{"status":"ok",...}`.

**3. Frontend on Vercel**
1. **Add New → Project** → import the same repository.
2. Set **Root Directory** to `services/webapp`. Vercel detects Vite (build `npm run build`, output `dist`).
3. Add the environment variable `VITE_API_BASE_URL` = your Render URL, `https://<your-service>.onrender.com` (no trailing slash). It is applied at build time, so redeploy after changing it.

**4. Connect them**: in Render, set `CHAT_CORS_ORIGINS` to your Vercel URL, e.g. `https://your-app.vercel.app` (no trailing slash, `https`; several origins can be comma-separated). Render redeploys automatically. Each Vercel *preview* URL is a different origin and must be added too if you want to use it.

**Free-tier behaviour to expect**
- The service sleeps after about 15 minutes without traffic and needs up to a minute to wake. The frontend pings `/api/health` when the page opens and retries a failed first message for roughly 70 seconds, showing "Connecting to the server… it may be waking up".
- Render may restart a free service at any time. Running jobs and the replay log live in memory, so a restart during an answer loses that answer; the user just asks again. Conversation history lives in the browser, so it survives.
- Free-tier terms change; check Render's pricing page before relying on them.
- **There is no login, but new questions are rate limited** (next section). That protects your NVIDIA quota; it does not identify users, and the counters live in memory, so a restart resets them.

### Rate limiting

Every new question starts an agent run that spends NVIDIA quota, so `POST /api/chat` is limited:

- **Per client IP:** 6 new questions a minute and 60 an hour, and one running question at a time (a stopped run does not count).
- **All clients together:** 300 an hour. This backstop cannot be dodged by faking an address.
- Going over returns HTTP 429 with a `Retry-After` header, and the chat shows the reason ("Too many questions from your network. Please wait N seconds and try again."). Resuming a dropped stream (same `request_id`) is never counted.
- Change or switch off any limit with the `RATE_LIMIT_*` variables in the table above.

**Which address is "the client"?** Behind Render's proxy the connection address is the proxy's, so without help all visitors would share one allowance. `render.yaml` sets `CLIENT_IP_HEADER=cf-connecting-ip`. Render does not document that header, so check it on your own deployment: open the app, ask one question, and read the Render log. The line `chat accepted ip=203.0.113.9 via=cf-connecting-ip` should show your public IP. If it says `via=peer`, the header is absent: try `CLIENT_IP_HEADER=x-forwarded-for` (with `TRUSTED_PROXY_HOPS=1` it uses the last entry, the one Render appended), or rely on the global limit. The leftmost `X-Forwarded-For` value is never used, because Render does not remove one sent by a client, so it can be forged. If clients can set the named header themselves, a determined visitor can dodge the per-IP limit but never the global one.

## Extending the UI catalog

1. Add a component definition to `services/api/catalog.json`.
2. Add its shape, bounds, and safety validation in `services/api/energy_agent/genui.py`.
3. Add conversion to the A2UI-style `updateComponents` list in `ui_tree_to_a2ui`.
4. Implement the matching React component in `services/webapp/src/renderer.tsx`.
5. Add validation and rendering tests. Do not add generated executable JS or arbitrary HTML.


## Reference links

- A2UI protocol: https://github.com/a2ui-project/a2ui/blob/main/specification/v1_0/docs/a2ui_protocol.md
- A2UI renderer guide: https://github.com/a2ui-project/a2ui/blob/main/docs/public/guides/renderer-development.md
- A2UI client setup: https://github.com/a2ui-project/a2ui/blob/main/docs/public/guides/client-setup.md
- React: https://react.dev/
- Vite: https://vite.dev/
