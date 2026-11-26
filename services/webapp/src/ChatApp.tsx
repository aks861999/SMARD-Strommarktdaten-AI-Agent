import { FormEvent, KeyboardEvent, ChangeEvent, useEffect, useLayoutEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ArchitectureBanner from "./ArchitectureBanner";
import { ChatSurface } from "./renderer";
import type { A2UIMessage, ChatMessage, UIEvent } from "./types";

// Local-development default when VITE_API_BASE_URL is not set. A deployment sets it at build time (on Vercel:
// the backend's https address). An EMPTY value means "same origin": the Docker setup serves the page and
// proxies /api from one address, so requests are relative. Hence `??` and not `||`.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
// Conversation memory is always on: the last 10 question/answer exchanges are sent
// with every request. Must match MAX_HISTORY_TURNS in energy_agent/agent.py.
const HISTORY_TURNS = 10;
// Starter suggestions only prefill the input. They are deliberately short, varied
// in intent and free of fixed dates, so they never act as the app's "one true
// question" and never steer the agent: the backend knows the current date.
// The first starter is intentionally a multi-step request: it makes the agent chain several
// tool calls (prices, statistics, cheapest window, two cost estimates) before it can answer.
// `summary` is the short card text; `prompt` is what gets prefilled.
const STARTER_PROMPTS: { title: string; summary?: string; prompt: string }[] = [
  {
    title: "Detailed price analysis",
    summary: "Multi-step: find the cheapest 3-hour window, price it, and compare it with an evening run.",
    prompt: "Tomorrow I need to run a 2 kW appliance for exactly 3 consecutive hours. Retrieve the available hourly prices, calculate the daily average, minimum and maximum, find the cheapest 3-hour window during the daytime, estimate the wholesale cost of that window, compare it with running it during the evening peak and recommend the cheaper option. Do not invent missing prices.",
  },
  { title: "Price pattern", prompt: "How do day-ahead prices look across the day tomorrow?" },
  { title: "Cheapest window", prompt: "When is the cheapest 2-hour window tomorrow?" },
  { title: "Run cost", prompt: "What would it cost to run a 1.5 kW appliance for 2 hours starting at 10:00 tomorrow?" },
];
const uid = () => crypto.randomUUID();
// Reconnect budget: with the 0.5 s -> 8 s backoff below, 12 retries wait roughly 70 s in
// total, enough for a sleeping free-tier backend (about a minute) to wake up.
const MAX_RETRIES = 12;

class FatalRequestError extends Error {}

// On a phone or tablet the on-screen keyboard's Enter key should add a new line (the send button
// sends), as in most chat apps; with a mouse and keyboard Enter sends and Shift+Enter adds a line.
const isTouchDevice = () => typeof window !== "undefined" && window.matchMedia("(pointer: coarse)").matches;

function traceTitle(event: UIEvent): string {
  switch (event.type) {
    case "agent_iteration": return `============ AGENT ITERATION ${event.step ?? ""} ============`;
    case "agent_decision": return "[AGENT DECISION] LLM response:";
    case "agent_final_response": return "[AGENT FINAL] LLM response:";
    case "tool_start": return `[AGENT DECISION] Call tool: ${event.name ?? "tool"}`;
    case "tool_result": return `[TOOL OBSERVATION] ${event.name ?? "tool"}`;
    case "agent_final": return "[AGENT] Final answer generated.";
    case "agent_limit": return "[AGENT] Iteration/runtime safety limit reached.";
    default: return event.type;
  }
}

function formatLLMResponse(event: UIEvent): string {
  // Keep the same field order as the user's LLMResponse execution-log example.
  return JSON.stringify({
    content: event.content ?? "",
    tool_calls: event.tool_calls ?? [],
    raw_assistant_message: event.raw_assistant_message ?? {
      role: "assistant",
      content: event.content || null,
    },
  }, null, 2);
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(new DOMException("Aborted", "AbortError")); return; }
    const timer = window.setTimeout(resolve, ms);
    signal.addEventListener("abort", () => {
      window.clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    }, { once: true });
  });
}

export default function ChatApp() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef<string | null>(null);
  // Resolved slots from the previous turn (date, duration, power, …), used by the
  // backend to resolve follow-ups such as "make it four hours instead".
  const turnStateRef = useRef<Record<string, unknown> | null>(null);

  // Auto-grow the message box with its content. The CSS max-height (140px) caps it, after
  // which the box scrolls. Runs on every input change, so it also shrinks back to one
  // line after sending and grows when a starter card prefills a long prompt.
  useLayoutEffect(() => {
    const box = textareaRef.current;
    if (!box) return;
    box.style.height = "auto";
    box.style.height = `${box.scrollHeight}px`;
  }, [input]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, busy]);
  useEffect(() => () => controllerRef.current?.abort(), []);

  // Silent wake-up call: a free-tier backend sleeps when idle and needs up to a minute to
  // start. Pinging it as soon as the page opens usually warms it before the first question.
  useEffect(() => { void fetch(`${API_BASE}/api/health`).catch(() => undefined); }, []);

  function updateAssistant(id: string, patch: Partial<ChatMessage>) {
    setMessages((prev) => prev.map((m) => m.id === id ? { ...m, ...patch } : m));
  }

  async function sendMessage(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    setError("");
    setInput("");

    const userMessage: ChatMessage = { id: uid(), role: "user", content: trimmed };
    const assistantId = uid();
    const requestId = uid();
    const assistantMessage: ChatMessage = {
      id: assistantId, role: "assistant", content: "", uiMessages: [],
      status: "Starting…", tools: [], trace: [],
    };
    // Always send the last HISTORY_TURNS exchanges (2 messages each).
    const priorHistory = messages
      .filter((m) => (m.role === "user" || m.role === "assistant") && m.content.trim())
      .slice(-HISTORY_TURNS * 2)
      .map(({ role, content }) => ({ role, content }));

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setBusy(true);
    const controller = new AbortController();
    controllerRef.current = controller;
    requestIdRef.current = requestId;

    let accumulatedText = "";
    let accumulatedUi: A2UIMessage[] = [];
    let lastEventId = 0;
    let terminal = false;
    let terminalError = "";
    let retries = 0;

    const appendTrace = (event: UIEvent) => {
      setMessages((prev) => prev.map((m) => m.id === assistantId
        ? { ...m, trace: [...(m.trace || []), event] }
        : m));
    };

    const handleEvent = (event: UIEvent) => {
      switch (event.type) {
        case "text_delta":
          // Progressive delivery: the text answer is shown the moment it exists; the
          // optional UI, which needs a second model call, is appended afterwards.
          accumulatedText += event.delta || "";
          setMessages((prev) => prev.map((m) => m.id === assistantId
            ? { ...m, content: m.content + (event.delta || "") }
            : m));
          break;
        case "status":
          updateAssistant(assistantId, { status: event.status || "Working…" });
          break;
        case "agent_iteration":
        case "agent_decision":
        case "agent_final_response":
        case "agent_final":
        case "agent_limit":
          appendTrace(event);
          break;
        case "tool_start":
          appendTrace(event);
          setMessages((prev) => prev.map((m) => m.id === assistantId
            ? { ...m, tools: [...(m.tools || []), event.name || "tool"], status: `Using ${event.name}…` }
            : m));
          break;
        case "tool_result":
          appendTrace(event);
          updateAssistant(assistantId, { status: `Received result from ${event.name}` });
          break;
        case "a2ui_message":
          if (event.message) {
            accumulatedUi = [...accumulatedUi, event.message];
            setMessages((prev) => prev.map((m) => m.id === assistantId
              ? { ...m, uiMessages: [...(m.uiMessages || []), event.message!] }
              : m));
          }
          break;
        case "ui_error":
          updateAssistant(assistantId, { status: "Text answer ready · optional UI could not be generated", error: event.error || "UI generation failed." });
          break;
        case "done":
          terminal = true;
          // Carry resolved slots (date, duration, power, …) into the next turn.
          if (event.state && typeof event.state === "object") {
            turnStateRef.current = event.state as Record<string, unknown>;
          }
          updateAssistant(assistantId, {
            content: event.answer || accumulatedText,
            uiMessages: accumulatedUi,
            status: "Complete",
          });
          break;
        case "error":
          terminal = true;
          terminalError = event.error || "Unknown agent error.";
          updateAssistant(assistantId, { status: "Error", error: terminalError });
          break;
      }
    };

    const consumeSSE = async (response: Response) => {
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`Server error ${response.status}: ${detail.slice(0, 500)}`);
      }
      if (!response.body) throw new Error("Streaming response body is unavailable.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      const processFrame = (frame: string) => {
        const lines = frame.split(/\r?\n/);
        const idLine = lines.find((line) => line.startsWith("id:"));
        if (idLine) {
          const parsedId = Number(idLine.slice(3).trim());
          if (Number.isFinite(parsedId)) lastEventId = Math.max(lastEventId, parsedId);
        }
        const data = lines.filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).replace(/^ /, "")).join("\n");
        if (!data) return;
        try { handleEvent(JSON.parse(data) as UIEvent); } catch { /* Ignore malformed event frames. */ }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
        let boundary: number;
        while ((boundary = buffer.indexOf("\n\n")) >= 0) {
          processFrame(buffer.slice(0, boundary));
          buffer = buffer.slice(boundary + 2);
        }
      }
      buffer += decoder.decode().replace(/\r\n/g, "\n");
      if (buffer.trim()) processFrame(buffer);
    };

    try {
      // `jobCreated` flips only after the server accepted the POST. Until then a retry must
      // POST again (safe: the same request_id + payload resumes, never duplicates, the job).
      // Resuming through the events endpoint before the job exists would just 404, which is
      // what happened when a sleeping free-tier server was still waking up.
      let jobCreated = false;
      while (!terminal && !controller.signal.aborted) {
        try {
          const response = !jobCreated
            ? await fetch(`${API_BASE}/api/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
                body: JSON.stringify({
                  message: trimmed, history: priorHistory,
                  state: turnStateRef.current ?? undefined, request_id: requestId,
                }),
                signal: controller.signal,
              })
            : await fetch(`${API_BASE}/api/chat/${encodeURIComponent(requestId)}/events?after=${lastEventId}`, {
                headers: { Accept: "text/event-stream" }, signal: controller.signal,
              });
          // A definite client-side rejection (bad request, unknown/expired job) will not
          // improve by retrying; 408/429 and every 5xx (e.g. a waking server) are retried.
          if (response.status >= 400 && response.status < 500 && response.status !== 408) {
            const detail = await response.text();
            let message = detail.slice(0, 500);
            try { const parsed = JSON.parse(detail); if (typeof parsed.detail === "string") message = parsed.detail; } catch { /* not JSON */ }
            // 429 = rate limited. Show the server's plain-language reason and do not retry in a loop.
            throw new FatalRequestError(response.status === 429 ? message : `Server error ${response.status}: ${message}`);
          }
          if (response.ok) jobCreated = true;
          await consumeSSE(response);
          if (!terminal) {
            retries += 1;
            if (retries > MAX_RETRIES) throw new Error("The agent stream could not be resumed after repeated connection interruptions.");
            await sleep(Math.min(500 * (2 ** (retries - 1)), 8000), controller.signal);
          }
        } catch (err) {
          if ((err as Error).name === "AbortError" || err instanceof FatalRequestError) throw err;
          retries += 1;
          if (retries > MAX_RETRIES) throw err;
          if (!jobCreated) updateAssistant(assistantId, { status: "Connecting to the server… it may be waking up, retrying" });
          await sleep(Math.min(500 * (2 ** (retries - 1)), 8000), controller.signal);
        }
      }

      if (!terminal && !controller.signal.aborted) {
        updateAssistant(assistantId, {
          status: "Incomplete",
          error: "The agent stream ended before its final event.",
        });
      }
      if (terminalError) setError(terminalError);
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        updateAssistant(assistantId, {
          content: accumulatedText || undefined,
          status: "Stopped by user",
        });
      } else {
        const msg = (err as Error).message || String(err);
        updateAssistant(assistantId, { error: msg, status: "Error" });
        setError(msg);
      }
    } finally {
      controllerRef.current = null;
      requestIdRef.current = null;
      setBusy(false);
    }
  }

  function stopGeneration() {
    const requestId = requestIdRef.current;
    if (requestId) {
      void fetch(`${API_BASE}/api/chat/${encodeURIComponent(requestId)}/cancel`, {
        method: "POST", keepalive: true,
      }).catch(() => undefined);
    }
    controllerRef.current?.abort();
  }

  function submit(event: FormEvent) { event.preventDefault(); void sendMessage(input); }

  function clear() {
    stopGeneration();
    setMessages([]);
    setError("");
    setBusy(false);
    turnStateRef.current = null;
  }

  return <div className="app-shell">
    <aside className="side-panel">
      <div className="brand"><span className="brand-icon">⚡</span><span>Energy Agent</span></div>
      <button className="new-chat" onClick={clear}>＋ New chat</button>
      <div className="side-info"><span className="live-dot"/> Live SMARD day-ahead prices<br/>DE-LU bidding zone</div>
      <div className="side-info">Remembers your last {HISTORY_TURNS} questions and answers in this chat.</div>
      <div className="side-foot">Chat-native dynamic UI<br/>A2UI · trusted component catalog</div>
    </aside>
    <main className="chat-main">
      <header className="topbar">
        <div className="topbar-title">Energy Price Planning Agent</div>
        <div className="topbar-right">
          <span className="model-tag">GenUI · streaming</span>
          {/* Phones hide the sidebar, so "New chat" lives here. Hidden on wide screens by CSS. */}
          <button className="topbar-new" onClick={clear} aria-label="Start a new chat">＋ New</button>
        </div>
      </header>
      <section className="conversation">
        {messages.length === 0 ? <div className="welcome">
          <ArchitectureBanner />
          <div className="welcome-icon">⚡</div><h1>What can I help you<br/>understand today?</h1>
          <p>Ask about energy prices, operating windows, costs or market data.</p>
          <div className="example-grid">
            {STARTER_PROMPTS.map((starter) => <button key={starter.title} className="example-card" onClick={() => setInput(starter.prompt)}><span className="example-title">{starter.title}</span><span className="example-text">{starter.summary ?? starter.prompt}</span><span className="example-arrow">↗</span></button>)}
          </div>
        </div> : messages.map((m) => <article className={`message ${m.role}`} key={m.id}>
          <div className="message-avatar">{m.role === "user" ? "U" : "⚡"}</div>
          <div className="message-content">
            {m.role === "assistant" ? <>
              {m.content && <div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown></div>}
              {!m.content && busy && <div className="thinking"><span className="pulse"/> {m.status || "Thinking…"}</div>}
              {m.uiMessages?.length ? <ChatSurface messages={m.uiMessages} onAsk={(prompt) => void sendMessage(prompt)} /> : null}
              {m.status && busy && <div className="agent-status">{m.status}</div>}
              {m.error && <div className="inline-error">{m.error}</div>}
              {m.tools?.length ? <details className="tool-details"><summary>Tools used ({m.tools.length})</summary><div>{m.tools.join(" · ")}</div></details> : null}
              {(m.trace?.length || (busy && !m.content)) ? <details className="agent-trace">
                <summary>Behind the scenes · Agent execution ({m.trace?.length || 0} events)</summary>
                {m.trace?.length ? <ol className="trace-list">{m.trace.map((event, index) => <li key={`${event.type}-${event.step ?? ""}-${index}`}>
                  <strong>{traceTitle(event)}</strong>
                  {event.type === "agent_decision" || event.type === "agent_final_response"
                    ? <>
                        <small>LLMResponse · content, tool_calls, raw_assistant_message</small>
                        <pre>{formatLLMResponse(event)}</pre>
                      </>
                    : null}
                  {event.type === "tool_start" ? <>
                    <small>[TOOL ARGUMENTS]</small>
                    <pre>{JSON.stringify(event.arguments ?? {}, null, 2)}</pre>
                    {event.deduplicated ? <small>Cached result reused for this exact call.</small> : null}
                  </> : null}
                  {event.type === "tool_result" && event.result !== undefined ? <>
                    <small>[TOOL OBSERVATION]</small>
                    <pre>{JSON.stringify(event.result, null, 2)}</pre>
                  </> : null}
                  {event.type === "agent_limit" && event.error ? <pre>{event.error}</pre> : null}
                </li>)}</ol> : <div className="trace-empty">Agent trace will appear as the agent makes decisions and calls tools.</div>}
              </details> : null}
            </> : <div className="user-text">{m.content}</div>}
          </div>
        </article>)}
        <div ref={bottomRef}/>
      </section>
      {error && <div className="error-toast" role="alert">{error}<button onClick={() => setError("")}>Dismiss</button></div>}
      <form className="composer-wrap" onSubmit={submit}>
        <div className="composer">
          <textarea ref={textareaRef} rows={1} enterKeyHint={isTouchDevice() ? "enter" : "send"} value={input} onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setInput(e.target.value)} placeholder="Ask about electricity prices…" disabled={busy} onKeyDown={(e: KeyboardEvent<HTMLTextAreaElement>) => { if (e.key === "Enter" && !e.shiftKey && !isTouchDevice()) { e.preventDefault(); void sendMessage(input); } }}/>
          {busy ? <button type="button" className="stop-btn" onClick={stopGeneration} title="Stop response">■</button> : <button type="submit" className="send-btn" disabled={!input.trim()} title="Send message">↑</button>}
        </div>
        <div className="composer-caption">AI can make mistakes. Market cost estimates are wholesale only and exclude retail charges.</div>
      </form>
    </main>
  </div>;
}
