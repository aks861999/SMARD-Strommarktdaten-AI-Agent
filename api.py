"""FastAPI SSE bridge with replayable agent events and inline GenUI surfaces."""
from __future__ import annotations

import asyncio
import hashlib
import json
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
from energy_agent.sources import DemoPriceSource, SmardPriceSource
from energy_agent.tools import EnergyTools


app = FastAPI(title="Energy Agent · Chat GenUI", version="2.1.0")

_MAX_ACTIVE = max(1, int(os.getenv("MAX_CONCURRENT_AGENT_RUNS", "2")))
_AGENT_SEMAPHORE = threading.Semaphore(_MAX_ACTIVE)
_origins = [
    x.strip()
    for x in os.getenv(
        "CHAT_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if x.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Last-Event-ID"],
)


class ChatRequest(BaseModel):
    # The LLM provider is deliberately NOT a client choice: it is backend
    # configuration (LLM_PROVIDER, default "nvidia"). Unknown fields are ignored.
    message: str = Field(min_length=1, max_length=8000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)
    state: dict[str, Any] | None = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1, max_length=80)


class AgentJob:
    """In-memory job event log so a client can reconnect and replay missed SSE events."""

    def __init__(self, request_id: str, request_signature: str = ""):
        self.request_id = request_id
        self.request_signature = request_signature
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


def _get_or_create_job(request_id: str, signature: str = "") -> tuple[AgentJob, bool]:
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
        job = AgentJob(request_id, signature)
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
    source = (
        DemoPriceSource()
        if os.getenv("PRICE_SOURCE", "smard").lower() == "demo"
        else SmardPriceSource()
    )
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

    job, created = _get_or_create_job(payload.request_id, _request_signature(payload))
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
