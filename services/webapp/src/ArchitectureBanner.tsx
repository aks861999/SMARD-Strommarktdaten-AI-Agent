import { Fragment, useState } from "react";

// A static, high-level picture of how the pieces connect. It is pure presentation: no model output,
// no dates, no example questions (see "Rule #1" in CLAUDE.md). The collapsed view is a 5-block row; the
// expanded "technical details" view shows services, the agent loop and progressive delivery.
//
// TOOLS must list the names exposed in services/api/energy_agent/tools.py (TOOL_SCHEMAS). A backend test
// (test_architecture_banner.py) fails when they drift apart.
const BLOCKS: { icon: string; title: string; caption: string }[] = [
  { icon: "💬", title: "You", caption: "Ask in plain language" },
  { icon: "🖥️", title: "Chat app", caption: "Shows answers, charts and steps" },
  { icon: "🧠", title: "Planning agent", caption: "Decides which tools to use" },
  { icon: "⚡", title: "Price service", caption: "Hourly day-ahead prices (EUR/MWh)" },
  { icon: "📊", title: "SMARD data", caption: "Real DE-LU market data" },
];

const TOOLS = ["get_hourly_prices", "analyze_prices", "find_cheapest_window", "estimate_run_cost"];

const SERVICES: { name: string; role: string; lines: string[]; out?: string }[] = [
  { name: "webapp", role: "React chat UI", lines: ["nginx: static files + /api proxy", "the only published service"] },
  { name: "api", role: "FastAPI + SSE", lines: ["agent loop · UI planner", "per-client rate limiter"], out: "LLM provider (backend only)" },
  { name: "prices", role: "HTTP price API", lines: ["TTL cache · no secrets", "never caches failures"], out: "SMARD live · demo CSV fallback" },
];

const LOOP = ["LLM decides", "Tool call", "Observation"];
const TEXT_LANE = ["Final answer", "Text stream", "Chat (shown first)"];
const UI_LANE = ["UI planner (JSON only)", "Validate (+1 repair retry)", "A2UI over SSE", "React catalog"];

const GUARDRAILS = [
  "Tool arguments validated; unknown ones rejected",
  "Duplicate tool calls cached",
  "Runtime limit and cancellation",
  "Per-client rate limit",
  "Bounded conversation history",
  "UI size and depth caps",
  "Numbers come only from tool results",
];

const DISMISS_KEY = "architecture-banner-dismissed";
const DETAILS_KEY = "architecture-banner-details";

// localStorage can throw (private window, blocked site data), so every access is guarded and the
// banner simply falls back to its defaults (visible, collapsed).
function readFlag(key: string): boolean {
  try { return window.localStorage.getItem(key) === "1"; } catch { return false; }
}
function writeFlag(key: string, value: boolean) {
  try {
    if (value) window.localStorage.setItem(key, "1");
    else window.localStorage.removeItem(key);
  } catch { /* storage unavailable: the choice just won't persist */ }
}

function Lane({ label, steps, loop }: { label?: string; steps: string[]; loop?: string }) {
  return (
    <div className="arch-lane-wrap">
      {label && <span className="arch-lane-label">{label}</span>}
      <ol className="arch-lane">
        {steps.map((step, i) => (
          <Fragment key={step}>
            {i > 0 && <li className="arch-step-arrow" aria-hidden="true" />}
            <li className="arch-step">{step}</li>
          </Fragment>
        ))}
      </ol>
      {loop && <span className="arch-loop-note"><span aria-hidden="true">↻ </span>{loop}</span>}
    </div>
  );
}

export default function ArchitectureBanner() {
  const [dismissed, setDismissed] = useState(() => readFlag(DISMISS_KEY));
  const [detailed, setDetailed] = useState(() => readFlag(DETAILS_KEY));

  function setDismiss(value: boolean) { setDismissed(value); writeFlag(DISMISS_KEY, value); }
  function setDetails(value: boolean) { setDetailed(value); writeFlag(DETAILS_KEY, value); }

  if (dismissed) {
    return <button type="button" className="arch-reopen" onClick={() => setDismiss(false)}>ⓘ How it works</button>;
  }

  return (
    <section className="arch-banner" aria-labelledby="arch-title">
      <div className="arch-head">
        <h2 id="arch-title">How this assistant works</h2>
        <div className="arch-head-actions">
          <button type="button" className="arch-toggle" aria-expanded={detailed} aria-controls="arch-details" onClick={() => setDetails(!detailed)}>
            {detailed ? "Hide technical details ▴" : "Show technical details ▾"}
          </button>
          <button type="button" className="arch-close" onClick={() => setDismiss(true)} aria-label="Hide the overview">✕</button>
        </div>
      </div>

      {!detailed && (
        <>
          <ol className="arch-flow">
            {BLOCKS.map((block, i) => (
              <Fragment key={block.title}>
                {i > 0 && <li className="arch-arrow" aria-hidden="true" />}
                <li className="arch-node">
                  <span className="arch-icon" aria-hidden="true">{block.icon}</span>
                  <span className="arch-title">{block.title}</span>
                  <span className="arch-caption">{block.caption}</span>
                </li>
              </Fragment>
            ))}
          </ol>
          <p className="arch-return"><span aria-hidden="true">↩ </span>The agent sends the text answer back first, then optional charts and tables.</p>
        </>
      )}

      {detailed && (
        <div id="arch-details" className="arch-details">
          <section className="arch-panel" aria-labelledby="arch-p1">
            <h3 id="arch-p1"><span aria-hidden="true">① </span>Services <small>docker compose, one network</small></h3>
            <ol className="arch-services">
              {SERVICES.map((s, i) => (
                <Fragment key={s.name}>
                  {i > 0 && <li className="arch-arrow" aria-hidden="true" />}
                  <li className="arch-service">
                    <span className="arch-svc-name">{s.name}</span>
                    <span className="arch-svc-role">{s.role}</span>
                    {s.lines.map((line) => <span className="arch-caption" key={line}>{line}</span>)}
                    {s.out && <span className="arch-out"><span aria-hidden="true">↳ </span>{s.out}</span>}
                  </li>
                </Fragment>
              ))}
            </ol>
          </section>

          <section className="arch-panel" aria-labelledby="arch-p2">
            <h3 id="arch-p2"><span aria-hidden="true">② </span>Agent loop <small>tool calling, no framework</small></h3>
            <Lane steps={LOOP} loop="repeats until no tool is needed" />
            <div className="arch-chips" aria-label="Tools">
              {TOOLS.map((tool) => <code className="arch-chip" key={tool}>{tool}</code>)}
            </div>
            <p className="arch-note">Stops on: no further tool call · runtime limit · optional step ceiling · cancellation.</p>
          </section>

          <section className="arch-panel" aria-labelledby="arch-p3">
            <h3 id="arch-p3"><span aria-hidden="true">③ </span>Delivery <small>progressive: text first, UI after</small></h3>
            <Lane label="Answer" steps={TEXT_LANE} />
            <Lane label="Optional UI" steps={UI_LANE} />
          </section>

          <section className="arch-panel" aria-labelledby="arch-p4">
            <h3 id="arch-p4">Guardrails</h3>
            <ul className="arch-guards">
              {GUARDRAILS.map((g) => <li key={g}>{g}</li>)}
            </ul>
            <p className="arch-note">Every decision and observation is visible in the chat under “Behind the scenes”.</p>
          </section>
        </div>
      )}
    </section>
  );
}
