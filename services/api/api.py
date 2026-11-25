"""FastAPI SSE bridge with replayable agent events and inline GenUI surfaces."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from main import build_llm
from energy_agent.agent import MAX_HISTORY_MESSAGES, Agent
from energy_agent.ratelimit import SlidingWindowLimiter, client_ip
from energy_agent.sources import HttpPriceSource, local_price_source
from energy_agent.tools import EnergyTools

# "uvicorn.error" is the logger uvicorn configures, so these lines show up in the host's log (Render).
logger = logging.getLogger("uvicorn.error")


app = FastAPI(title="Energy Agent · Chat GenUI", version="2.1.0")

_MAX_ACTIVE = max(1, int(os.getenv("MAX_CONCURRENT_AGENT_RUNS", "2")))
_AGENT_SEMAPHORE = threading.Semaphore(_MAX_ACTIVE)
# Browser origins allowed to call this API. The default below is for local development only (the Vite dev
# server). A deployed backend sets CHAT_CORS_ORIGINS to its real frontend address (see render.yaml and the
# README); if it is forgotten the app fails closed, refusing every browser except a local one.
_DEV_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
_origins = [x.strip() for x in os.getenv("CHAT_CORS_ORIGINS", _DEV_ORIGINS).split(",") if x.strip()]
if not os.getenv("CHAT_CORS_ORIGINS") and os.getenv("RENDER"):  # Render sets RENDER=true on its services
    logger.warning("CHAT_CORS_ORIGINS is not set on Render: only local origins are allowed, so the deployed "
                   "frontend will be blocked by CORS. Set it to your frontend address.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Last-Event-ID"],
)


# --- Rate limiting -------------------------------------------------------------------------------
# Every new question starts an agent run that spends LLM quota, so new questions are limited per client
# IP and, as a backstop that cannot be dodged by forging an address, across all clients. Resuming a
# stream with a request_id that already exists is never counted. A value of 0 switches a limit off.
#   RATE_LIMIT_PER_MINUTE (6)   RATE_LIMIT_PER_HOUR (60)   RATE_LIMIT_GLOBAL_PER_HOUR (300)
#   RATE_LIMIT_MAX_ACTIVE_PER_IP (1): runs one client may have going at once; a cancelled run does not count
#   CLIENT_IP_HEADER (unset): header holding the visitor IP behind a proxy, e.g. cf-connecting-ip
#   TRUSTED_PROXY_HOPS (1): only for CLIENT_IP_HEADER=x-forwarded-for, see energy_agent/ratelimit.py
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _build_rate_limiters() -> tuple[SlidingWindowLimiter, SlidingWindowLimiter]:
    per_ip = SlidingWindowLimiter([(_env_int("RATE_LIMIT_PER_MINUTE", 6), 60),
                                   (_env_int("RATE_LIMIT_PER_HOUR", 60), 3600)])
    everyone = SlidingWindowLimiter([(_env_int("RATE_LIMIT_GLOBAL_PER_HOUR", 300), 3600)])
    return per_ip, everyone


_IP_LIMITER, _GLOBAL_LIMITER = _build_rate_limiters()


def _reset_rate_limits() -> None:
    """Rebuild the limiters from the environment (used by tests; also forgets all history)."""
    global _IP_LIMITER, _GLOBAL_LIMITER
    _IP_LIMITER, _GLOBAL_LIMITER = _build_rate_limiters()


def _reject(ip: str, source: str, kind: str, wait: float, message: str) -> None:
    seconds = max(1, math.ceil(wait))
    logger.warning("chat rejected (%s) ip=%s via=%s retry_after=%ss", kind, ip, source, seconds)
    raise HTTPException(status_code=429, detail=message, headers={"Retry-After": str(seconds)})


class ChatRequest(BaseModel):
    # The LLM provider is deliberately NOT a client choice: it is backend
    # configuration (LLM_PROVIDER, default "nvidia"). Unknown fields are ignored.
    message: str = Field(min_length=1, max_length=8000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)
    state: dict[str, Any] | None = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1, max_length=80)


class AgentJob:
    """In-memory job event log so a client can reconnect and replay missed SSE events."""

    def __init__(self, request_id: str, request_signature: str = "", client_ip: str = ""):
        self.request_id = request_id
        self.request_signature = request_signature
        self.client_ip = client_ip
        self.events: list[tuple[int, dict[str, Any]]] = []
        self.sequence = 0
        self.condition = threading.Condition()
        self.cancel_event = threading.Event()
        self.finished = False
        self.created_at = time.time()

    def publish(self, event: dict[str, Any]) -> None:
        with self.condition:
            self.sequence += 1
            self.events.append((self.sequence, event))
            self.condition.notify_all()

    def finish(self) -> None:
        with self.condition:
            self.finished = True
            self.condition.notify_all()

    def read_after(self, cursor: int, timeout: float = 1.0):
        with self.condition:
            if not any(seq > cursor for seq, _ in self.events) and not self.finished:
                self.condition.wait(timeout)
            batch = [(seq, event) for seq, event in self.events if seq > cursor]
            return batch, self.finished, self.sequence


_JOBS: dict[str, AgentJob] = {}
_JOBS_LOCK = threading.Lock()
_JOB_TTL_SECONDS = 60 * 60
_MAX_RETAINED_JOBS = 128


def _prune_jobs() -> None:
    now = time.time()
    stale = [
        request_id for request_id, job in _JOBS.items()
        if job.finished and now - job.created_at > _JOB_TTL_SECONDS
    ]
    for request_id in stale:
        _JOBS.pop(request_id, None)
    if len(_JOBS) > _MAX_RETAINED_JOBS:
        finished = sorted(
            (job for job in _JOBS.values() if job.finished),
            key=lambda item: item.created_at,
        )
        for job in finished[: max(0, len(_JOBS) - _MAX_RETAINED_JOBS)]:
            _JOBS.pop(job.request_id, None)


def _request_signature(payload: ChatRequest) -> str:
    material = json.dumps(
        {"message": payload.message,
         "history": payload.history, "state": payload.state},
        sort_keys=True, default=str, ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _admit(ip: str, source: str) -> None:
    """Raise HTTP 429 unless this client may start another run. Call with _JOBS_LOCK held."""
    max_active = _env_int("RATE_LIMIT_MAX_ACTIVE_PER_IP", 1)
    if max_active > 0:
        active = sum(1 for job in _JOBS.values()
                     if job.client_ip == ip and not job.finished and not job.cancel_event.is_set())
        if active >= max_active:
            _reject(ip, source, "active", 5,
                    "Your previous question is still running. Wait for it to finish or press Stop, then ask again.")
    wait_ip = _IP_LIMITER.retry_after(ip)
    if wait_ip > 0:
        _reject(ip, source, "per-ip", wait_ip,
                f"Too many questions from your network. Please wait {max(1, math.ceil(wait_ip))} seconds and try again.")
    wait_all = _GLOBAL_LIMITER.retry_after("*")
    if wait_all > 0:
        _reject(ip, source, "global", wait_all, "The service is busy right now. Please try again in a few minutes.")
    # Every limiter agreed: spend budget from all of them (nothing is spent on a rejected request).
    _IP_LIMITER.record(ip)
    _GLOBAL_LIMITER.record("*")
    logger.info("chat accepted ip=%s via=%s", ip, source)


def _get_or_create_job(request_id: str, signature: str = "", client_ip: str = "",
                       admit=None) -> tuple[AgentJob, bool]:
    with _JOBS_LOCK:
        _prune_jobs()
        existing = _JOBS.get(request_id)
        if existing is not None:
            # The same request_id resumes its stream; a different payload reusing
            # an id would silently attach to someone else's answer and be cancellable.
            if existing.request_signature and signature and existing.request_signature != signature:
                raise HTTPException(
                    status_code=409,
                    detail="request_id is already used by a different request. Use a new request_id.",
                )
            return existing, False
        if admit is not None:
            admit()  # may raise 429; nothing has been created yet, so a rejected request leaves no trace
        job = AgentJob(request_id, signature, client_ip)
        _JOBS[request_id] = job
        return job, True


def _get_job(request_id: str) -> AgentJob | None:
    with _JOBS_LOCK:
        _prune_jobs()
        return _JOBS.get(request_id)


def make_agent() -> Agent:
    from types import SimpleNamespace

    provider = os.getenv("LLM_PROVIDER", "nvidia").strip().lower()
    if provider not in {"nvidia", "llama-cpp"}:
        raise ValueError(f"LLM_PROVIDER must be 'nvidia' or 'llama-cpp', got '{provider}'.")
    args = SimpleNamespace(provider=provider, model=None, base_url=None)
    llm = build_llm(args)
    prices_url = os.getenv("PRICES_URL", "").strip()
    if prices_url:  # docker compose: prices come from the separate, caching prices service
        source = HttpPriceSource(prices_url)
    else:  # single-service run (Render, local development): the SMARD / demo code runs in this process
        source = local_price_source(os.getenv("PRICE_SOURCE", "smard").strip().lower())
    raw_max_steps = int(os.getenv("AGENT_MAX_STEPS", "0"))
    raw_runtime = int(os.getenv("AGENT_MAX_RUNTIME_SECONDS", "900"))
    return Agent(
        llm=llm,
        tools=EnergyTools(price_source=source),
        max_steps=raw_max_steps or None,
        max_runtime_seconds=raw_runtime or None,
        verbose=False,
    )


def _encode(sequence: int, event: dict[str, Any]) -> str:
    return f"id: {sequence}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


def _start_worker(job: AgentJob, payload: ChatRequest) -> None:
    def worker() -> None:
        acquired = False
        try:
            job.publish({"type": "status", "status": "Waiting for an available agent slot…"})
            while not job.cancel_event.is_set():
                acquired = _AGENT_SEMAPHORE.acquire(timeout=0.5)
                if acquired:
                    break
            if job.cancel_event.is_set():
                job.publish({"type": "status", "status": "Stopped by user"})
                return
            agent = make_agent()
            answer, ui_messages = agent.run(
                payload.message.strip(),
                conversation_context=payload.history,
                prior_state=payload.state,
                on_event=job.publish,
                cancel_event=job.cancel_event,
            )
            job.publish({"type": "done", "answer": answer, "has_ui": bool(ui_messages),
                         "state": getattr(agent, "last_turn_state", None)})
        except Exception as exc:
            job.publish({"type": "error", "error": str(exc)})
        finally:
            if acquired:
                _AGENT_SEMAPHORE.release()
            job.finish()

    threading.Thread(target=worker, daemon=True, name=f"agent-{job.request_id[:8]}").start()


async def _job_event_stream(request: Request, job: AgentJob, after: int = 0):
    cursor = max(0, after)
    last_keepalive = time.monotonic()
    while True:
        if await request.is_disconnected():
            # Do not cancel the job on a transient/disposable SSE connection loss.
            # The browser can reconnect to the event log using its last event id.
            break
        batch, finished, latest = await asyncio.to_thread(job.read_after, cursor, 1.0)
        if batch:
            for sequence, event in batch:
                cursor = sequence
                yield _encode(sequence, event)
        elif time.monotonic() - last_keepalive >= 10:
            yield ": keep-alive\n\n"
            last_keepalive = time.monotonic()
        if finished and cursor >= latest:
            break


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "energy-chat-genui"}


@app.get("/api/catalog")
def catalog() -> dict:
    catalog_path = os.path.join(os.path.dirname(__file__), "catalog.json")
    with open(catalog_path, encoding="utf-8") as file:
        return json.load(file)


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request):
    query = payload.message.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")
    if len(payload.history) > MAX_HISTORY_MESSAGES:
        raise HTTPException(status_code=422,
                            detail=f"History is limited to {MAX_HISTORY_MESSAGES} messages.")

    ip, source = client_ip(request.headers, request.client.host if request.client else None,
                           os.getenv("CLIENT_IP_HEADER", ""), _env_int("TRUSTED_PROXY_HOPS", 1))
    job, created = _get_or_create_job(payload.request_id, _request_signature(payload), ip,
                                      admit=lambda: _admit(ip, source))
    if created:
        _start_worker(job, payload)
    return StreamingResponse(
        _job_event_stream(request, job),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/chat/{request_id}/events")
async def reconnect_chat(request_id: str, request: Request, after: int = Query(default=0, ge=0)):
    job = _get_job(request_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Chat job not found or expired.")
    return StreamingResponse(
        _job_event_stream(request, job, after=after),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/chat/{request_id}/cancel")
def cancel_chat(request_id: str) -> dict[str, str]:
    job = _get_job(request_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Chat job not found or expired.")
    job.cancel_event.set()
    return {"status": "cancellation_requested", "request_id": request_id}
